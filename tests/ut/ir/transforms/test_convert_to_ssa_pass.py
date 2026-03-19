# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for ConvertToSSA pass.

Tests use the Before/Expected pattern with @pl.program decorator.
Uses assert_structural_equal with enable_auto_mapping=True to compare.
"""

import pypto.language as pl
import pytest
from pypto import DataType, ir, passes
from pypto.language.parser.diagnostics import SSAViolationError

# =============================================================================
# Category 1: Straight-line Code with Structural Equality
# =============================================================================


class TestStraightLineCode:
    """Tests for straight-line code with multiple assignments."""

    def test_single_reassignment(self):
        """result = add(x, 1); result = add(result, 2) -> result_0, result_1"""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                result = pl.add(x, 1.0)
                result = pl.add(result, 2.0)
                return result

        @pl.program
        class Expected:
            @pl.function(strict_ssa=True)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                result_0: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
                result_1: pl.Tensor[[64], pl.FP32] = pl.add(result_0, 2.0)
                return result_1

        After = passes.convert_to_ssa()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_multiple_reassignments(self):
        """result = ...; result = ...; result = ... -> result_0, result_1, result_2"""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                result = pl.add(x, 1.0)
                result = pl.add(result, 2.0)
                result = pl.add(result, 3.0)
                return result

        @pl.program
        class Expected:
            @pl.function(strict_ssa=True)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                result_0: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
                result_1: pl.Tensor[[64], pl.FP32] = pl.add(result_0, 2.0)
                result_2: pl.Tensor[[64], pl.FP32] = pl.add(result_1, 3.0)
                return result_2

        After = passes.convert_to_ssa()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_reassignment_with_self_reference(self):
        """result = mul(x, 2); result = add(result, x) -> uses previous version on RHS"""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                result = pl.mul(x, 2.0)
                result = pl.add(result, x)
                return result

        @pl.program
        class Expected:
            @pl.function(strict_ssa=True)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                result_0: pl.Tensor[[64], pl.FP32] = pl.mul(x, 2.0)
                result_1: pl.Tensor[[64], pl.FP32] = pl.add(result_0, x)
                return result_1

        After = passes.convert_to_ssa()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_multiple_variables(self):
        """a = ...; b = ...; a = ...; b = ... -> a_0, a_1, b_0, b_1"""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                a = pl.add(x, 1.0)
                b = pl.mul(x, 2.0)
                a = pl.add(a, 3.0)
                b = pl.mul(b, 4.0)
                result = pl.add(a, b)
                return result

        @pl.program
        class Expected:
            @pl.function(strict_ssa=True)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                a_0: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
                b_0: pl.Tensor[[64], pl.FP32] = pl.mul(x, 2.0)
                a_1: pl.Tensor[[64], pl.FP32] = pl.add(a_0, 3.0)
                b_1: pl.Tensor[[64], pl.FP32] = pl.mul(b_0, 4.0)
                result_0: pl.Tensor[[64], pl.FP32] = pl.add(a_1, b_1)
                return result_0

        After = passes.convert_to_ssa()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_already_ssa_no_reassignment(self):
        """a = ...; b = ... -> a_0, b_0 (versioned but no conflicts)"""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                a = pl.add(x, 1.0)
                b = pl.mul(a, 2.0)
                return b

        @pl.program
        class Expected:
            @pl.function(strict_ssa=True)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                a_0: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
                b_0: pl.Tensor[[64], pl.FP32] = pl.mul(a_0, 2.0)
                return b_0

        After = passes.convert_to_ssa()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_parameter_versioning(self):
        """Parameters should get version suffixes (x -> x_0)."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                result = pl.add(x, 1.0)
                return result

        @pl.program
        class Expected:
            @pl.function(strict_ssa=True)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                result_0: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
                return result_0

        After = passes.convert_to_ssa()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_already_ssa_is_unchanged(self):
        """Already-SSA code should be unchanged after conversion."""

        @pl.program
        class Before:
            @pl.function(strict_ssa=True)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                a: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
                b: pl.Tensor[[64], pl.FP32] = pl.mul(a, 2.0)
                return b

        After = passes.convert_to_ssa()(Before)
        ir.assert_structural_equal(After, Before)


# =============================================================================
# Category 2: For Loops with Structural Equality
# =============================================================================


class TestForLoops:
    """Tests for for loop conversion to SSA with iter_args."""

    def test_loop_with_iter_args(self):
        """for loop with iter_args should be preserved with versioned names."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                init = pl.create_tensor([64], dtype=pl.FP32)
                for i, (acc,) in pl.range(10, init_values=(init,)):
                    new_acc = pl.add(acc, x)
                    result = pl.yield_(new_acc)
                return result

        @pl.program
        class Expected:
            @pl.function(strict_ssa=True)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                init_0: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                for i_0, (acc_0,) in pl.range(10, init_values=(init_0,)):
                    new_acc_0: pl.Tensor[[64], pl.FP32] = pl.add(acc_0, x)
                    result_0 = pl.yield_(new_acc_0)
                return result_0

        After = passes.convert_to_ssa()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_loop_with_multiple_iter_args(self):
        """for loop with multiple iter_args should preserve all of them."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                init1: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                init2: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                for i, (acc1, acc2) in pl.range(5, init_values=(init1, init2)):
                    new1: pl.Tensor[[64], pl.FP32] = pl.add(acc1, x)
                    new2: pl.Tensor[[64], pl.FP32] = pl.mul(acc2, 2.0)
                    out1, out2 = pl.yield_(new1, new2)
                result: pl.Tensor[[64], pl.FP32] = pl.add(out1, out2)
                return result

        @pl.program
        class Expected:
            @pl.function(strict_ssa=True)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                init1_0: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                init2_0: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                for i_0, (acc1_0, acc2_0) in pl.range(5, init_values=(init1_0, init2_0)):
                    new1_0: pl.Tensor[[64], pl.FP32] = pl.add(acc1_0, x)
                    new2_0: pl.Tensor[[64], pl.FP32] = pl.mul(acc2_0, 2.0)
                    out1_0, out2_0 = pl.yield_(new1_0, new2_0)
                result_0: pl.Tensor[[64], pl.FP32] = pl.add(out1_0, out2_0)
                return result_0

        After = passes.convert_to_ssa()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_loop_with_range_params(self):
        """for loop with start, stop, step parameters."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                init: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                for i, (acc,) in pl.range(0, 10, 2, init_values=(init,)):
                    new_acc: pl.Tensor[[64], pl.FP32] = pl.add(acc, x)
                    result = pl.yield_(new_acc)
                return result

        @pl.program
        class Expected:
            @pl.function(strict_ssa=True)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                init_0: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                for i_0, (acc_0,) in pl.range(0, 10, 2, init_values=(init_0,)):
                    new_acc_0: pl.Tensor[[64], pl.FP32] = pl.add(acc_0, x)
                    result_0 = pl.yield_(new_acc_0)
                return result_0

        After = passes.convert_to_ssa()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_nested_for_loops(self):
        """Nested for loops."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                init: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                for i, (outer,) in pl.range(3, init_values=(init,)):
                    for j, (inner,) in pl.range(2, init_values=(outer,)):
                        new_inner: pl.Tensor[[64], pl.FP32] = pl.add(inner, 1.0)
                        inner_out = pl.yield_(new_inner)
                    outer_out = pl.yield_(inner_out)
                return outer_out

        @pl.program
        class Expected:
            @pl.function(strict_ssa=True)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                init_0: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                for i_0, (outer_0,) in pl.range(3, init_values=(init_0,)):
                    for j_0, (inner_0,) in pl.range(2, init_values=(outer_0,)):
                        new_inner_0: pl.Tensor[[64], pl.FP32] = pl.add(inner_0, 1.0)
                        inner_out_0 = pl.yield_(new_inner_0)
                    outer_out_0 = pl.yield_(inner_out_0)
                return outer_out_0

        After = passes.convert_to_ssa()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_sequential_loops(self):
        """Sequential (not nested) for loops."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                init: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                for i, (acc,) in pl.range(5, init_values=(init,)):
                    new_acc: pl.Tensor[[64], pl.FP32] = pl.add(acc, 1.0)
                    result1 = pl.yield_(new_acc)
                for j, (acc2,) in pl.range(3, init_values=(result1,)):
                    new_acc2: pl.Tensor[[64], pl.FP32] = pl.mul(acc2, 2.0)
                    result2 = pl.yield_(new_acc2)
                return result2

        @pl.program
        class Expected:
            @pl.function(strict_ssa=True)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                init_0: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                for i_0, (acc_0,) in pl.range(5, init_values=(init_0,)):
                    new_acc_0: pl.Tensor[[64], pl.FP32] = pl.add(acc_0, 1.0)
                    result1_0 = pl.yield_(new_acc_0)
                for j_0, (acc2_0,) in pl.range(3, init_values=(result1_0,)):
                    new_acc2_0: pl.Tensor[[64], pl.FP32] = pl.mul(acc2_0, 2.0)
                    result2_0 = pl.yield_(new_acc2_0)
                return result2_0

        After = passes.convert_to_ssa()(Before)
        ir.assert_structural_equal(After, Expected)


# =============================================================================
# Category 3: While Loops
# =============================================================================


class TestWhileLoops:
    """Tests for while loop conversion to SSA form with iter_args."""

    def test_simple_while_loop(self):
        """while x < n: x = x + 1 -> for x_iter in pl.while_(x_iter < n, init_values=(x_0,))"""

        @pl.program
        class Before:
            @pl.function
            def main(self, n: pl.Scalar[pl.INT64]) -> pl.Scalar[pl.INT64]:
                x: pl.Scalar[pl.INT64] = 0
                while x < n:
                    x = x + 1
                return x

        @pl.program
        class Expected:
            @pl.function(strict_ssa=True)
            def main(self, n_0: pl.Scalar[pl.INT64]) -> pl.Scalar[pl.INT64]:
                x_0: pl.Scalar[pl.INT64] = 0
                for (x_iter_1,) in pl.while_(init_values=(x_0,)):
                    pl.cond(x_iter_1 < n_0)
                    x_3: pl.Scalar[pl.INT64] = x_iter_1 + 1
                    x_2 = pl.yield_(x_3)
                return x_2

        After = passes.convert_to_ssa()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_while_loop_multiple_variables(self):
        """while loop with multiple loop-carried variables"""

        @pl.program
        class Before:
            @pl.function
            def main(self, n: pl.Scalar[pl.INT64]) -> pl.Scalar[pl.INT64]:
                x: pl.Scalar[pl.INT64] = 0
                y: pl.Scalar[pl.INT64] = 1
                while x < n:
                    x = x + 1
                    y = y * 2
                return y

        @pl.program
        class Expected:
            @pl.function(strict_ssa=True)
            def main(self, n_0: pl.Scalar[pl.INT64]) -> pl.Scalar[pl.INT64]:
                x_0: pl.Scalar[pl.INT64] = 0
                y_0: pl.Scalar[pl.INT64] = 1
                for x_iter_1, y_iter_1 in pl.while_(init_values=(x_0, y_0)):
                    pl.cond(x_iter_1 < n_0)
                    x_3: pl.Scalar[pl.INT64] = x_iter_1 + 1
                    y_3: pl.Scalar[pl.INT64] = y_iter_1 * 2
                    x_2, y_2 = pl.yield_(x_3, y_3)
                return y_2

        After = passes.convert_to_ssa()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_nested_while_loops(self):
        """Nested while loops -> both converted to SSA"""

        @pl.program
        class Before:
            @pl.function
            def main(self, n: pl.Scalar[pl.INT64]) -> pl.Scalar[pl.INT64]:
                x: pl.Scalar[pl.INT64] = 0
                while x < n:
                    y: pl.Scalar[pl.INT64] = 0
                    while y < 3:
                        y = y + 1
                    x = x + 1
                return x

        @pl.program
        class Expected:
            @pl.function(strict_ssa=True)
            def main(self, n_0: pl.Scalar[pl.INT64]) -> pl.Scalar[pl.INT64]:
                x_0: pl.Scalar[pl.INT64] = 0
                for (x_iter_1,) in pl.while_(init_values=(x_0,)):
                    pl.cond(x_iter_1 < n_0)
                    y_0: pl.Scalar[pl.INT64] = 0
                    for (y_iter_1,) in pl.while_(init_values=(y_0,)):
                        pl.cond(y_iter_1 < 3)
                        y_3: pl.Scalar[pl.INT64] = y_iter_1 + 1
                        y_2 = pl.yield_(y_3)  # noqa: F841
                    x_3: pl.Scalar[pl.INT64] = x_iter_1 + 1
                    x_2 = pl.yield_(x_3)
                return x_2

        After = passes.convert_to_ssa()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_while_in_for_loop(self):
        """While loop nested inside for loop"""

        @pl.program
        class Before:
            @pl.function
            def main(self, n: pl.Scalar[pl.INT64]) -> pl.Scalar[pl.INT64]:
                init_sum: pl.Scalar[pl.INT64] = 0
                for i, (sum_val,) in pl.range(5, init_values=(init_sum,)):
                    x: pl.Scalar[pl.INT64] = 0
                    while x < i:
                        x = x + 1
                    new_sum: pl.Scalar[pl.INT64] = sum_val + x
                    sum_out = pl.yield_(new_sum)
                return sum_out

        @pl.program
        class Expected:
            @pl.function(strict_ssa=True)
            def main(self, n_0: pl.Scalar[pl.INT64]) -> pl.Scalar[pl.INT64]:
                init_sum_0: pl.Scalar[pl.INT64] = 0
                for i_0, (sum_val,) in pl.range(0, 5, 1, init_values=(init_sum_0,)):
                    x_0: pl.Scalar[pl.INT64] = 0
                    for (x_iter_1,) in pl.while_(init_values=(x_0,)):
                        pl.cond(x_iter_1 < i_0)
                        x_3: pl.Scalar[pl.INT64] = x_iter_1 + 1
                        x_2 = pl.yield_(x_3)
                    new_sum_0: pl.Scalar[pl.INT64] = sum_val + x_2
                    sum_val_out = pl.yield_(new_sum_0)
                return sum_val_out

        After = passes.convert_to_ssa()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_for_in_while_loop(self):
        """For loop nested inside while loop"""

        @pl.program
        class Before:
            @pl.function
            def main(self, n: pl.Scalar[pl.INT64]) -> pl.Scalar[pl.INT64]:
                x: pl.Scalar[pl.INT64] = 0
                while x < n:
                    init_acc: pl.Scalar[pl.INT64] = x
                    for i, (acc,) in pl.range(3, init_values=(init_acc,)):
                        new_acc: pl.Scalar[pl.INT64] = acc + 1
                        acc_out = pl.yield_(new_acc)
                    x = acc_out
                return x

        @pl.program
        class Expected:
            @pl.function(strict_ssa=True)
            def main(self, n_0: pl.Scalar[pl.INT64]) -> pl.Scalar[pl.INT64]:
                x_0: pl.Scalar[pl.INT64] = 0
                for (x_iter_1,) in pl.while_(init_values=(x_0,)):
                    pl.cond(x_iter_1 < n_0)
                    init_acc_0: pl.Scalar[pl.INT64] = x_iter_1
                    for i_0, (acc,) in pl.range(0, 3, 1, init_values=(init_acc_0,)):
                        new_acc_0: pl.Scalar[pl.INT64] = acc + 1
                        acc_out = pl.yield_(new_acc_0)
                    x_3: pl.Scalar[pl.INT64] = acc_out
                    x_2 = pl.yield_(x_3)
                return x_2

        After = passes.convert_to_ssa()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_sequential_while_loops(self):
        """Sequential while loops with shared variables"""

        @pl.program
        class Before:
            @pl.function
            def main(self, n: pl.Scalar[pl.INT64]) -> pl.Scalar[pl.INT64]:
                x: pl.Scalar[pl.INT64] = 0
                while x < n:
                    x = x + 1
                y: pl.Scalar[pl.INT64] = x
                while y < 10:
                    y = y + 2
                return y

        @pl.program
        class Expected:
            @pl.function(strict_ssa=True)
            def main(self, n: pl.Scalar[pl.INT64]) -> pl.Scalar[pl.INT64]:
                x_0: pl.Scalar[pl.INT64] = 0
                for (x_1,) in pl.while_(init_values=(x_0,)):
                    pl.cond(x_1 < n)
                    x_2: pl.Scalar[pl.INT64] = x_1 + 1
                    x_3 = pl.yield_(x_2)
                y_0: pl.Scalar[pl.INT64] = x_3
                for (y_1,) in pl.while_(init_values=(y_0,)):
                    pl.cond(y_1 < 10)
                    y_2: pl.Scalar[pl.INT64] = y_1 + 2
                    y_3 = pl.yield_(y_2)
                return y_3

        After = passes.convert_to_ssa()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_while_with_tensor_operations(self):
        """While loop with tensor operations"""

        @pl.program
        class Before:
            @pl.function
            def main(self, n: pl.Scalar[pl.INT64], x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                i: pl.Scalar[pl.INT64] = 0
                acc: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                while i < n:
                    i = i + 1
                    acc = pl.add(acc, x)
                return acc

        @pl.program
        class Expected:
            @pl.function(strict_ssa=True)
            def main(self, n: pl.Scalar[pl.INT64], x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                i_0: pl.Scalar[pl.INT64] = 0
                acc_0: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                for acc_1, i_1 in pl.while_(init_values=(acc_0, i_0)):
                    pl.cond(i_1 < n)
                    i_2: pl.Scalar[pl.INT64] = i_1 + 1
                    acc_2: pl.Tensor[[64], pl.FP32] = pl.add(acc_1, x)
                    acc_3, i_3 = pl.yield_(acc_2, i_2)
                return acc_3

        After = passes.convert_to_ssa()(Before)
        ir.assert_structural_equal(After, Expected)


# =============================================================================
# Category 4: If Statements (inside loops, since if needs scalar condition)
# =============================================================================


class TestIfStatements:
    """Tests for if statement conversion to SSA with phi nodes."""

    def test_if_in_loop_both_branches(self):
        """if cond: val=mul(...) else: val=add(...) -> phi node for val"""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                init: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                for i, (acc,) in pl.range(5, init_values=(init,)):
                    if i == 0:
                        val = pl.mul(acc, 2.0)
                        out = pl.yield_(val)
                    else:
                        val2 = pl.add(acc, x)
                        out = pl.yield_(val2)
                    result = pl.yield_(out)
                return result

        @pl.program
        class Expected:
            @pl.function(strict_ssa=True)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                init_0: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                for i_0, (acc_0,) in pl.range(5, init_values=(init_0,)):
                    if i_0 == 0:
                        val_0 = pl.mul(acc_0, 2.0)
                        out_0 = pl.yield_(val_0)
                    else:
                        val2_0 = pl.add(acc_0, x)
                        out_0 = pl.yield_(val2_0)
                    result_0 = pl.yield_(out_0)
                return result_0

        After = passes.convert_to_ssa()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_if_in_loop_then_only(self):
        """if cond: new_val = mul(...) else: yield acc -> phi with pre-if value"""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                init: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                for i, (acc,) in pl.range(3, init_values=(init,)):
                    if i == 0:
                        new_acc: pl.Tensor[[64], pl.FP32] = pl.mul(acc, 2.0)
                        val = pl.yield_(new_acc)
                    else:
                        val = pl.yield_(acc)
                    result = pl.yield_(val)
                return result

        @pl.program
        class Expected:
            @pl.function(strict_ssa=True)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                init_0: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                for i_0, (acc_0,) in pl.range(3, init_values=(init_0,)):
                    if i_0 == 0:
                        new_acc_0: pl.Tensor[[64], pl.FP32] = pl.mul(acc_0, 2.0)
                        val_0 = pl.yield_(new_acc_0)
                    else:
                        val_0 = pl.yield_(acc_0)
                    result_0 = pl.yield_(val_0)
                return result_0

        After = passes.convert_to_ssa()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_multiple_vars_modified_in_if(self):
        """if cond: a=..; b=.. else: a=..; b=.. -> phi for both a and b"""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                init1: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                init2: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                for i, (a, b) in pl.range(5, init_values=(init1, init2)):
                    if i == 0:
                        new_a: pl.Tensor[[64], pl.FP32] = pl.mul(a, 2.0)
                        new_b: pl.Tensor[[64], pl.FP32] = pl.mul(b, 3.0)
                        out_a, out_b = pl.yield_(new_a, new_b)
                    else:
                        out_a, out_b = pl.yield_(a, b)
                    res_a, res_b = pl.yield_(out_a, out_b)
                result: pl.Tensor[[64], pl.FP32] = pl.add(res_a, res_b)
                return result

        @pl.program
        class Expected:
            @pl.function(strict_ssa=True)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                init1_0: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                init2_0: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                for i_0, (a_0, b_0) in pl.range(5, init_values=(init1_0, init2_0)):
                    if i_0 == 0:
                        new_a_0: pl.Tensor[[64], pl.FP32] = pl.mul(a_0, 2.0)
                        new_b_0: pl.Tensor[[64], pl.FP32] = pl.mul(b_0, 3.0)
                        out_a_0, out_b_0 = pl.yield_(new_a_0, new_b_0)
                    else:
                        out_a_0, out_b_0 = pl.yield_(a_0, b_0)
                    res_a_0, res_b_0 = pl.yield_(out_a_0, out_b_0)
                result_0: pl.Tensor[[64], pl.FP32] = pl.add(res_a_0, res_b_0)
                return result_0

        After = passes.convert_to_ssa()(Before)
        ir.assert_structural_equal(After, Expected)


# =============================================================================
# Category 4: strict_ssa=True Mode (Parser Tests)
# =============================================================================


class TestStrictSSAMode:
    """Tests for strict_ssa=True enforcement in the parser."""

    def test_strict_ssa_single_assignment_passes(self):
        """SSA-compliant code should pass with strict_ssa=True."""
        # Note: strict_ssa must be on @pl.program, not @pl.function (inner decorator doesn't execute)

        @pl.program(strict_ssa=True)
        class ValidSSA:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                result: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
                return result

        assert ValidSSA is not None

    def test_strict_ssa_multiple_assignment_fails(self):
        """Multiple assignments should fail with strict_ssa=True."""
        # Note: strict_ssa must be on @pl.program, not @pl.function (inner decorator doesn't execute)
        with pytest.raises(SSAViolationError):

            @pl.program(strict_ssa=True)
            class InvalidSSA:
                @pl.function
                def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                    result = pl.add(x, 1.0)
                    result = pl.add(result, 2.0)
                    return result

    def test_non_strict_ssa_allows_reassignment(self):
        """Multiple assignments should succeed with strict_ssa=False (default)."""

        @pl.program
        class NonSSAFunc:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                result = pl.add(x, 1.0)
                result = pl.add(result, 2.0)
                return result

        assert NonSSAFunc is not None


# =============================================================================
# Category 5: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and corner scenarios."""

    def test_reserved_auto_name_delimiter_in_base_raises(self):
        """Base names containing '__' should be rejected before auto-naming."""

        @pl.program
        class Before:
            @pl.function
            def main(self, bad__x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                result: pl.Tensor[[64], pl.FP32] = pl.add(bad__x, bad__x)
                return result

        with pytest.raises(ValueError, match="reserved delimiter '__'"):
            passes.convert_to_ssa()(Before)

    def test_variables_with_numeric_suffixes(self):
        """Variables ending in _<digits> should be treated as distinct (issue #170)."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                tmp_0: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                tmp_1: pl.Tensor[[64], pl.FP32] = pl.add(tmp_0, x)
                result: pl.Tensor[[64], pl.FP32] = pl.add(tmp_1, tmp_0)
                return result

        @pl.program
        class Expected:
            @pl.function(strict_ssa=True)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                tmp_0_0: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                tmp_1_0: pl.Tensor[[64], pl.FP32] = pl.add(tmp_0_0, x)
                result_0: pl.Tensor[[64], pl.FP32] = pl.add(tmp_1_0, tmp_0_0)
                return result_0

        After = passes.convert_to_ssa()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_multiple_params(self):
        """Function with multiple parameters all get versioned."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                x: pl.Tensor[[64], pl.FP32],
                y: pl.Tensor[[64], pl.FP32],
                z: pl.Tensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                result = pl.add(x, y)
                result = pl.add(result, z)
                return result

        @pl.program
        class Expected:
            @pl.function(strict_ssa=True)
            def main(
                self,
                x: pl.Tensor[[64], pl.FP32],
                y: pl.Tensor[[64], pl.FP32],
                z: pl.Tensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                result_0: pl.Tensor[[64], pl.FP32] = pl.add(x, y)
                result_1: pl.Tensor[[64], pl.FP32] = pl.add(result_0, z)
                return result_1

        After = passes.convert_to_ssa()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_unused_variable(self):
        """Unused variable should still be versioned."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                unused: pl.Tensor[[64], pl.FP32] = pl.mul(x, 3.0)  # noqa: F841
                result: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
                return result

        @pl.program
        class Expected:
            @pl.function(strict_ssa=True)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                unused_0: pl.Tensor[[64], pl.FP32] = pl.mul(x, 3.0)  # noqa: F841
                result_0: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
                return result_0

        After = passes.convert_to_ssa()(Before)
        ir.assert_structural_equal(After, Expected)


# =============================================================================
# Plain Syntax Tests (without pl.yield_ and with simple for loop)
# =============================================================================


class TestPlainSyntax:
    """Tests for plain Python-like syntax without explicit pl.yield_() and iter_args."""

    def test_simple_for_loop_plain(self):
        """Simple for i in pl.range(n) converting to iter_args pattern."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                acc: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                for i in pl.range(10):
                    acc = pl.add(acc, 1.0)
                return acc

        @pl.program
        class Expected:
            @pl.function(strict_ssa=True)
            def main(self, x_0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                acc_0: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                for i_0, (acc_iter_1,) in pl.range(0, 10, 1, init_values=(acc_0,)):
                    acc_2: pl.Tensor[[64], pl.FP32] = pl.add(acc_iter_1, 1.0)
                    acc_1 = pl.yield_(acc_2)
                return acc_1

        After = passes.convert_to_ssa()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_for_loop_modifying_outer_var_plain(self):
        """For loop modifies variable defined before the loop."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                result: pl.Tensor[[64], pl.FP32] = x
                for i in pl.range(5):
                    result = pl.add(result, 1.0)
                return result

        @pl.program
        class Expected:
            @pl.function(strict_ssa=True)
            def main(self, x_0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                result_0: pl.Tensor[[64], pl.FP32] = x_0
                for i_0, (result_iter_1,) in pl.range(0, 5, 1, init_values=(result_0,)):
                    result_2: pl.Tensor[[64], pl.FP32] = pl.add(result_iter_1, 1.0)
                    result_1 = pl.yield_(result_2)
                return result_1

        After = passes.convert_to_ssa()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_for_loop_multiple_vars_modified_plain(self):
        """For loop modifies multiple outer variables."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                a: pl.Tensor[[64], pl.FP32] = x
                b: pl.Tensor[[64], pl.FP32] = pl.mul(x, 2.0)
                for i in pl.range(3):
                    a = pl.add(a, 1.0)
                    b = pl.mul(b, 1.5)
                result: pl.Tensor[[64], pl.FP32] = pl.add(a, b)
                return result

        @pl.program
        class Expected:
            @pl.function(strict_ssa=True)
            def main(self, x_0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                a_0: pl.Tensor[[64], pl.FP32] = x_0
                b_0: pl.Tensor[[64], pl.FP32] = pl.mul(x_0, 2.0)
                for i_0, (a_iter_1, b_iter_2) in pl.range(0, 3, 1, init_values=(a_0, b_0)):
                    a_3: pl.Tensor[[64], pl.FP32] = pl.add(a_iter_1, 1.0)
                    b_4: pl.Tensor[[64], pl.FP32] = pl.mul(b_iter_2, 1.5)
                    a_1, b_2 = pl.yield_(a_3, b_4)
                result_0: pl.Tensor[[64], pl.FP32] = pl.add(a_1, b_2)
                return result_0

        After = passes.convert_to_ssa()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_for_loop_no_outer_modification_plain(self):
        """For loop with only local assignments (no loop-carried variables needed)."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                for i in pl.range(5):
                    temp: pl.Tensor[[64], pl.FP32] = pl.mul(x, 2.0)
                    pl.add(temp, 1.0)
                return x

        @pl.program
        class Expected:
            @pl.function(strict_ssa=True)
            def main(self, x_0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                for i_0 in pl.range(0, 5, 1):
                    temp_0: pl.Tensor[[64], pl.FP32] = pl.mul(x_0, 2.0)
                    pl.add(temp_0, 1.0)
                return x_0

        After = passes.convert_to_ssa()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_nested_for_loops_plain(self):
        """Nested for loops with plain syntax."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                result: pl.Tensor[[64], pl.FP32] = x
                for i in pl.range(3):
                    for j in pl.range(2):
                        result = pl.add(result, 1.0)
                return result

        @pl.program
        class Expected:
            @pl.function(strict_ssa=True)
            def main(self, x_0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                result_0: pl.Tensor[[64], pl.FP32] = x_0
                for i_0, (result_iter_1,) in pl.range(0, 3, 1, init_values=(result_0,)):
                    for j_0, (result_iter_2,) in pl.range(0, 2, 1, init_values=(result_iter_1,)):
                        result_3: pl.Tensor[[64], pl.FP32] = pl.add(result_iter_2, 1.0)
                        result_2 = pl.yield_(result_3)
                    result_1 = pl.yield_(result_2)
                return result_1

        After = passes.convert_to_ssa()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_nested_for_loops_multiple_vars_plain(self):
        """Nested loops modifying multiple variables at different levels."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                outer: pl.Tensor[[64], pl.FP32] = x
                inner: pl.Tensor[[64], pl.FP32] = pl.mul(x, 2.0)
                for i in pl.range(2):
                    for j in pl.range(3):
                        inner = pl.add(inner, 1.0)
                    outer = pl.add(outer, inner)
                return outer

        @pl.program
        class Expected:
            @pl.function(strict_ssa=True)
            def main(self, x_0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                outer_0: pl.Tensor[[64], pl.FP32] = x_0
                inner_0: pl.Tensor[[64], pl.FP32] = pl.mul(x_0, 2.0)
                for i_0, (inner_iter_1, outer_iter_1) in pl.range(0, 2, 1, init_values=(inner_0, outer_0)):
                    for j_0, (inner_iter_3,) in pl.range(0, 3, 1, init_values=(inner_iter_1,)):
                        inner_5: pl.Tensor[[64], pl.FP32] = pl.add(inner_iter_3, 1.0)
                        inner_4 = pl.yield_(inner_5)
                    outer_3: pl.Tensor[[64], pl.FP32] = pl.add(outer_iter_1, inner_4)
                    inner_2, outer_2 = pl.yield_(inner_4, outer_3)
                return outer_2

        After = passes.convert_to_ssa()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_for_with_if_inside_plain(self):
        """For loop with if statement inside, both using plain syntax."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                result: pl.Tensor[[64], pl.FP32] = x
                for i in pl.range(5):
                    if i == 0:
                        result = pl.mul(result, 2.0)
                    else:
                        result = pl.add(result, 1.0)
                return result

        @pl.program
        class Expected:
            @pl.function(strict_ssa=True)
            def main(self, x_0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                result_0: pl.Tensor[[64], pl.FP32] = x_0
                for i_0, (result_iter_1,) in pl.range(0, 5, 1, init_values=(result_0,)):
                    if i_0 == 0:
                        result_3: pl.Tensor[[64], pl.FP32] = pl.mul(result_iter_1, 2.0)
                        result_5 = pl.yield_(result_3)
                    else:
                        result_4: pl.Tensor[[64], pl.FP32] = pl.add(result_iter_1, 1.0)
                        result_5 = pl.yield_(result_4)
                    result_2 = pl.yield_(result_5)
                return result_2

        After = passes.convert_to_ssa()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_nested_loops_with_if_plain(self):
        """Nested loops with if statement, all plain syntax."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                result: pl.Tensor[[64], pl.FP32] = x
                for i in pl.range(3):
                    for j in pl.range(2):
                        if j == 0:
                            result = pl.add(result, 1.0)
                        else:
                            result = pl.mul(result, 1.5)
                return result

        @pl.program
        class Expected:
            @pl.function(strict_ssa=True)
            def main(self, x_0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                result_0: pl.Tensor[[64], pl.FP32] = x_0
                for i_0, (result_iter_1,) in pl.range(0, 3, 1, init_values=(result_0,)):
                    for j_0, (result_iter_3,) in pl.range(0, 2, 1, init_values=(result_iter_1,)):
                        if j_0 == 0:
                            result_5: pl.Tensor[[64], pl.FP32] = pl.add(result_iter_3, 1.0)
                            result_7 = pl.yield_(result_5)
                        else:
                            result_6: pl.Tensor[[64], pl.FP32] = pl.mul(result_iter_3, 1.5)
                            result_7 = pl.yield_(result_6)
                        result_4 = pl.yield_(result_7)
                    result_2 = pl.yield_(result_4)
                return result_2

        After = passes.convert_to_ssa()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_complex_nested_control_flow_plain(self):
        """Complex nesting: for -> if -> for with multiple variables."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                a: pl.Tensor[[64], pl.FP32] = x
                b: pl.Tensor[[64], pl.FP32] = pl.mul(x, 2.0)
                for i in pl.range(2):
                    if i == 0:
                        for j in pl.range(2):
                            a = pl.add(a, 1.0)
                    else:
                        b = pl.mul(b, 2.0)
                result: pl.Tensor[[64], pl.FP32] = pl.add(a, b)
                return result

        @pl.program
        class Expected:
            @pl.function(strict_ssa=True)
            def main(self, x_0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                a_0: pl.Tensor[[64], pl.FP32] = x_0
                b_0: pl.Tensor[[64], pl.FP32] = pl.mul(x_0, 2.0)
                for i_0, (a_iter_1, b_iter_1) in pl.range(0, 2, 1, init_values=(a_0, b_0)):
                    if i_0 == 0:
                        for j_0, (a_iter_3,) in pl.range(0, 2, 1, init_values=(a_iter_1,)):
                            a_5: pl.Tensor[[64], pl.FP32] = pl.add(a_iter_3, 1.0)
                            a_4 = pl.yield_(a_5)
                        a_6, b_4 = pl.yield_(a_4, b_iter_1)
                    else:
                        b_3: pl.Tensor[[64], pl.FP32] = pl.mul(b_iter_1, 2.0)
                        a_6, b_4 = pl.yield_(a_iter_1, b_3)
                    a_2, b_2 = pl.yield_(a_6, b_4)
                result_0: pl.Tensor[[64], pl.FP32] = pl.add(a_2, b_2)
                return result_0

        After = passes.convert_to_ssa()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_multiple_sequential_loops_plain(self):
        """Multiple sequential loops using plain syntax."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                result: pl.Tensor[[64], pl.FP32] = x
                for i in pl.range(2):
                    result = pl.add(result, 1.0)
                for j in pl.range(3):
                    result = pl.mul(result, 1.5)
                return result

        @pl.program
        class Expected:
            @pl.function(strict_ssa=True)
            def main(self, x_0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                result_0: pl.Tensor[[64], pl.FP32] = x_0
                for i_0, (result_iter_1,) in pl.range(0, 2, 1, init_values=(result_0,)):
                    result_2: pl.Tensor[[64], pl.FP32] = pl.add(result_iter_1, 1.0)
                    result_1 = pl.yield_(result_2)
                for j_0, (result_iter_3,) in pl.range(0, 3, 1, init_values=(result_1,)):
                    result_4: pl.Tensor[[64], pl.FP32] = pl.mul(result_iter_3, 1.5)
                    result_3 = pl.yield_(result_4)
                return result_3

        After = passes.convert_to_ssa()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_if_modifying_different_vars_plain(self):
        """If statement where branches modify different variables."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                a: pl.Tensor[[64], pl.FP32] = x
                b: pl.Tensor[[64], pl.FP32] = pl.mul(x, 2.0)
                for i in pl.range(1):
                    if i == 0:
                        a = pl.add(a, 1.0)
                    else:
                        b = pl.add(b, 1.0)
                result: pl.Tensor[[64], pl.FP32] = pl.add(a, b)
                return result

        @pl.program
        class Expected:
            @pl.function(strict_ssa=True)
            def main(self, x_0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                a_0: pl.Tensor[[64], pl.FP32] = x_0
                b_0: pl.Tensor[[64], pl.FP32] = pl.mul(x_0, 2.0)
                for i_0, (a_iter_1, b_iter_1) in pl.range(0, 1, 1, init_values=(a_0, b_0)):
                    if i_0 == 0:
                        a_3: pl.Tensor[[64], pl.FP32] = pl.add(a_iter_1, 1.0)
                        a_4, b_4 = pl.yield_(a_3, b_iter_1)
                    else:
                        b_3: pl.Tensor[[64], pl.FP32] = pl.add(b_iter_1, 1.0)
                        a_4, b_4 = pl.yield_(a_iter_1, b_3)
                    a_2, b_2 = pl.yield_(a_4, b_4)
                result_0: pl.Tensor[[64], pl.FP32] = pl.add(a_2, b_2)
                return result_0

        After = passes.convert_to_ssa()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_plain_for_uses_outer_value_after_loop(self):
        """Variable modified in loop is accessible after loop."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                result: pl.Tensor[[64], pl.FP32] = x
                for i in pl.range(3):
                    result = pl.add(result, 1.0)
                final: pl.Tensor[[64], pl.FP32] = pl.mul(result, 2.0)
                return final

        @pl.program
        class Expected:
            @pl.function(strict_ssa=True)
            def main(self, x_0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                result_0: pl.Tensor[[64], pl.FP32] = x_0
                for i_0, (result_iter_1,) in pl.range(0, 3, 1, init_values=(result_0,)):
                    result_2: pl.Tensor[[64], pl.FP32] = pl.add(result_iter_1, 1.0)
                    result_1 = pl.yield_(result_2)
                final_0: pl.Tensor[[64], pl.FP32] = pl.mul(result_1, 2.0)
                return final_0

        After = passes.convert_to_ssa()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_if_with_empty_then_branch_plain(self):
        """Empty then-branch (e.g. from continue elimination) should not create single-child SeqStmts.

        Regression test for issue #561: ConvertToSSA was wrapping a single yield
        in a SeqStmts when inserting into an empty branch, violating NoRedundantBlocks.
        """

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                result: pl.Tensor[[64], pl.FP32] = x
                for i in pl.range(10):
                    if i > 5:
                        pass
                    else:
                        result = pl.add(result, 1.0)
                return result

        @pl.program
        class Expected:
            @pl.function(strict_ssa=True)
            def main(self, x_0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                result_0: pl.Tensor[[64], pl.FP32] = x_0
                for i_0, (result_iter_1,) in pl.range(0, 10, 1, init_values=(result_0,)):
                    if i_0 > 5:
                        result_4 = pl.yield_(result_iter_1)
                    else:
                        result_3: pl.Tensor[[64], pl.FP32] = pl.add(result_iter_1, 1.0)
                        result_4 = pl.yield_(result_3)
                    result_2 = pl.yield_(result_4)
                return result_2

        After = passes.convert_to_ssa()(Before)
        ir.assert_structural_equal(After, Expected)


class TestEscapingVariables:
    """Tests for variables first defined inside a loop but used after.

    The escaping-variable pattern occurs when a pass (e.g., ConvertTensorToTileOps)
    creates assignments inside a loop body for variables that are used after the loop.
    The DSL parser prevents this pattern, so tests build IR directly.
    """

    @staticmethod
    def _build_for_loop_escaping_program():
        """Build IR: for loop with `out` assigned inside, used in return after."""
        span = ir.Span.unknown()
        tensor_type = ir.TensorType([64], DataType.FP32)

        # Function params: a (In), c (Out)
        a = ir.Var("a", tensor_type, span)
        c = ir.Var("c", tensor_type, span)

        # Loop variable and range
        i = ir.Var("i", ir.ScalarType(DataType.INDEX), span)
        start = ir.ConstInt(0, DataType.INDEX, span)
        stop = ir.ConstInt(4, DataType.INDEX, span)
        step = ir.ConstInt(1, DataType.INDEX, span)

        # Body: out = add(a, c)  — 'out' is first defined HERE, inside the loop
        out = ir.Var("out", tensor_type, span)
        add_op = ir.Op("tensor.add")
        add_call = ir.Call(add_op, [a, c], span)
        body = ir.AssignStmt(out, add_call, span)

        # ForStmt with NO iter_args/return_vars (pre-SSA form)
        for_stmt = ir.ForStmt(i, start, stop, step, [], body, [], span)

        # return out — references variable defined inside loop
        ret = ir.ReturnStmt([out], span)

        func_body = ir.SeqStmts([for_stmt, ret], span)
        func = ir.Function(
            "main",
            [(a, ir.ParamDirection.In), (c, ir.ParamDirection.Out)],
            [tensor_type],
            func_body,
            span,
        )
        return ir.Program([func], "test", span)

    def test_for_loop_escaping_var(self):
        """Variable first assigned inside for loop, used after loop."""
        program = self._build_for_loop_escaping_program()
        after = passes.convert_to_ssa()(program)
        passes.run_verifier()(after)

    def test_for_loop_non_escaping_var_not_promoted(self):
        """Variable defined inside loop but NOT used after should not be promoted."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                result: pl.Tensor[[64], pl.FP32] = x
                for i in pl.range(4):
                    tmp: pl.Tensor[[64], pl.FP32] = pl.add(result, 1.0)
                    result = pl.add(tmp, 1.0)
                return result

        After = passes.convert_to_ssa()(Before)
        passes.run_verifier()(After)

    def test_dynamic_shape_vars_in_orchestrator(self):
        """Dynamic shape vars (M, N) from InCore return type don't cause scope violation."""
        M = pl.dynamic("M")
        N = pl.dynamic("N")

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def add_kernel(
                self,
                a: pl.Tensor[[M, N], pl.FP32],
                b: pl.Tensor[[M, N], pl.FP32],
                c: pl.Out[pl.Tensor[[M, N], pl.FP32]],
            ) -> pl.Tensor[[M, N], pl.FP32]:
                a_tile = pl.load(a, [0, 0], [128, 128], target_memory=pl.MemorySpace.Vec)
                b_tile = pl.load(b, [0, 0], [128, 128])
                result = pl.add(a_tile, b_tile)
                out = pl.store(result, [0, 0], c)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                b: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                c: pl.Tensor[[128, 128], pl.FP32] = pl.create_tensor([128, 128], dtype=pl.FP32)
                c = self.add_kernel(a, b, c)
                return c

        After = passes.convert_to_ssa()(Before)
        passes.run_verifier()(After)

    def test_nested_loop_local_temporaries_not_promoted(self):
        """Loop-local temporaries redefined in a subsequent loop must not escape.

        Regression test for issue #592: ConvertToSSA promoted loop-local
        temporaries (k0, x_chunk) into iter_args when the same names appeared
        in a subsequent loop, because UseCollector counted all recursive
        references without checking that the name was locally redefined.
        """

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                a: pl.Tensor[[64], pl.FP32] = x
                b: pl.Tensor[[64], pl.FP32] = pl.mul(x, 2.0)
                for i in pl.range(4):
                    tmp: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
                    a = pl.add(a, tmp)
                for j in pl.range(4):
                    tmp: pl.Tensor[[64], pl.FP32] = pl.add(x, 2.0)
                    b = pl.add(b, tmp)
                result: pl.Tensor[[64], pl.FP32] = pl.add(a, b)
                return result

        @pl.program
        class Expected:
            @pl.function(strict_ssa=True)
            def main(self, x_0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                a_0: pl.Tensor[[64], pl.FP32] = x_0
                b_0: pl.Tensor[[64], pl.FP32] = pl.mul(x_0, 2.0)
                for i_0, (a_iter_1,) in pl.range(0, 4, 1, init_values=(a_0,)):
                    tmp_0: pl.Tensor[[64], pl.FP32] = pl.add(x_0, 1.0)
                    a_2: pl.Tensor[[64], pl.FP32] = pl.add(a_iter_1, tmp_0)
                    a_1 = pl.yield_(a_2)
                for j_0, (b_iter_1,) in pl.range(0, 4, 1, init_values=(b_0,)):
                    tmp_1: pl.Tensor[[64], pl.FP32] = pl.add(x_0, 2.0)
                    b_2: pl.Tensor[[64], pl.FP32] = pl.add(b_iter_1, tmp_1)
                    b_1 = pl.yield_(b_2)
                result_0: pl.Tensor[[64], pl.FP32] = pl.add(a_1, b_1)
                return result_0

        After = passes.convert_to_ssa()(Before)
        ir.assert_structural_equal(After, Expected, enable_auto_mapping=True)

    def test_single_branch_escaping_var_gets_phi(self):
        """Variable defined only in one if-branch inside a loop must still escape.

        Regression test for issue #600: When CtrlFlowTransform lowers
        ``continue`` into an if/else with an empty branch, variables defined
        only in the else-branch were silently dropped by the IfStmt handler
        because the phi-node logic required both branches to define the
        variable. Pre-registering escaping vars as iter_args before body
        conversion ensures the IfStmt handler sees them in current_version_
        and creates a proper phi (pass-through in the empty branch, new
        value in the defining branch).
        """
        span = ir.Span.unknown()
        tensor_type = ir.TensorType([64], DataType.FP32)
        scalar_idx = ir.ScalarType(DataType.INDEX)

        x = ir.Var("x", tensor_type, span)
        c = ir.Var("c", tensor_type, span)
        i = ir.Var("i", scalar_idx, span)
        start = ir.ConstInt(0, DataType.INDEX, span)
        stop = ir.ConstInt(4, DataType.INDEX, span)
        step = ir.ConstInt(1, DataType.INDEX, span)

        # Condition: i % 2 != 0 (the continue path)
        two = ir.ConstInt(2, DataType.INDEX, span)
        zero_idx = ir.ConstInt(0, DataType.INDEX, span)
        mod_expr = ir.FloorMod(i, two, DataType.INDEX, span)
        cond = ir.Ne(mod_expr, zero_idx, DataType.BOOL, span)

        # else branch: out = add(x, c) — variable defined only here
        out = ir.Var("out", tensor_type, span)
        add_op = ir.Op("tensor.add")
        add_call = ir.Call(add_op, [x, c], span)
        else_body = ir.AssignStmt(out, add_call, span)

        # then branch: pass (empty — from continue lowering)
        then_body = ir.SeqStmts([], span)

        if_stmt = ir.IfStmt(cond, then_body, else_body, [], span)
        for_stmt = ir.ForStmt(i, start, stop, step, [], if_stmt, [], span)
        ret = ir.ReturnStmt([out], span)
        func_body = ir.SeqStmts([for_stmt, ret], span)

        func = ir.Function(
            "kernel",
            [(x, ir.ParamDirection.In), (c, ir.ParamDirection.Out)],
            [tensor_type],
            func_body,
            span,
            ir.FunctionType.InCore,
        )
        program = ir.Program([func], "test", span)

        after = passes.convert_to_ssa()(program)
        passes.run_verifier()(after)

    def test_loop_local_temporaries_in_model_pattern(self):
        """Real-world model pattern: k0/x_chunk loop-locals must not be promoted.

        Regression test for issue #601: identical root cause to #592 but
        reported from a production Qwen3 model. Variables like ``k0 = kb * K``
        and ``x_chunk = cast(slice(...))`` are pure loop-local temporaries
        recomputed each iteration — they must not appear in ``init_values``
        or ``yield_``.
        """

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                acc: pl.Tensor[[64], pl.FP32] = x
                for i in pl.range(4):
                    k0 = i * 8
                    chunk: pl.Tensor[[64], pl.FP32] = pl.add(x, k0)
                    acc = pl.add(acc, chunk)
                result: pl.Tensor[[64], pl.FP32] = pl.mul(acc, 2.0)
                return result

        After = passes.convert_to_ssa()(Before)

        # Verify loop-local temporaries are NOT promoted to iter_args
        printed = ir.python_print(After)
        assert "k0_iter" not in printed, "k0 should not be promoted to iter_arg"
        assert "chunk_iter" not in printed, "chunk should not be promoted to iter_arg"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
