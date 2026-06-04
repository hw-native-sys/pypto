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
Uses assert_structural_equal to compare. Both Before and Expected are complete
self-contained programs, so the default strict-identity mode (with DefField
auto-mapping at def sites) is sufficient.
"""

import pypto.language as pl
import pypto.language.distributed as pld
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
                result_0 = pl.add(x, 1.0)
                result_1 = pl.add(result_0, 2.0)
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
                result_0 = pl.add(x, 1.0)
                result_1 = pl.add(result_0, 2.0)
                result_2 = pl.add(result_1, 3.0)
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
                result_0 = pl.mul(x, 2.0)
                result_1 = pl.add(result_0, x)
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
                a_0 = pl.add(x, 1.0)
                b_0 = pl.mul(x, 2.0)
                a_1 = pl.add(a_0, 3.0)
                b_1 = pl.mul(b_0, 4.0)
                result_0 = pl.add(a_1, b_1)
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
                a_0 = pl.add(x, 1.0)
                b_0 = pl.mul(a_0, 2.0)
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
                result_0 = pl.add(x, 1.0)
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
                init_0 = pl.create_tensor([64], dtype=pl.FP32)
                for i_0, (acc_0,) in pl.range(10, init_values=(init_0,)):
                    new_acc_0 = pl.add(acc_0, x)
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
                init1_0 = pl.create_tensor([64], dtype=pl.FP32)
                init2_0 = pl.create_tensor([64], dtype=pl.FP32)
                for i_0, (acc1_0, acc2_0) in pl.range(5, init_values=(init1_0, init2_0)):
                    new1_0 = pl.add(acc1_0, x)
                    new2_0 = pl.mul(acc2_0, 2.0)
                    out1_0, out2_0 = pl.yield_(new1_0, new2_0)
                result_0 = pl.add(out1_0, out2_0)
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
                init_0 = pl.create_tensor([64], dtype=pl.FP32)
                for i_0, (acc_0,) in pl.range(0, 10, 2, init_values=(init_0,)):
                    new_acc_0 = pl.add(acc_0, x)
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
                init_0 = pl.create_tensor([64], dtype=pl.FP32)
                for i_0, (outer_0,) in pl.range(3, init_values=(init_0,)):
                    for j_0, (inner_0,) in pl.range(2, init_values=(outer_0,)):
                        new_inner_0 = pl.add(inner_0, 1.0)
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
                init_0 = pl.create_tensor([64], dtype=pl.FP32)
                for i_0, (acc_0,) in pl.range(5, init_values=(init_0,)):
                    new_acc_0 = pl.add(acc_0, 1.0)
                    result1_0 = pl.yield_(new_acc_0)
                for j_0, (acc2_0,) in pl.range(3, init_values=(result1_0,)):
                    new_acc2_0 = pl.mul(acc2_0, 2.0)
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
                    x_3 = x_iter_1 + 1
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
                    x_3 = x_iter_1 + 1
                    y_3 = y_iter_1 * 2
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
                        y_3 = y_iter_1 + 1
                        y_2 = pl.yield_(y_3)  # noqa: F841
                    x_3 = x_iter_1 + 1
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
                        x_3 = x_iter_1 + 1
                        x_2 = pl.yield_(x_3)
                    new_sum_0 = sum_val + x_2
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
                    init_acc_0 = x_iter_1
                    for i_0, (acc,) in pl.range(0, 3, 1, init_values=(init_acc_0,)):
                        new_acc_0 = acc + 1
                        acc_out = pl.yield_(new_acc_0)
                    x_3 = acc_out
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
                    x_2 = x_1 + 1
                    x_3 = pl.yield_(x_2)
                y_0 = x_3
                for (y_1,) in pl.while_(init_values=(y_0,)):
                    pl.cond(y_1 < 10)
                    y_2 = y_1 + 2
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
                acc_0 = pl.create_tensor([64], dtype=pl.FP32)
                for acc_1, i_1 in pl.while_(init_values=(acc_0, i_0)):
                    pl.cond(i_1 < n)
                    i_2 = i_1 + 1
                    acc_2 = pl.add(acc_1, x)
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
                init_0 = pl.create_tensor([64], dtype=pl.FP32)
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
                init_0 = pl.create_tensor([64], dtype=pl.FP32)
                for i_0, (acc_0,) in pl.range(3, init_values=(init_0,)):
                    if i_0 == 0:
                        new_acc_0 = pl.mul(acc_0, 2.0)
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
                init1_0 = pl.create_tensor([64], dtype=pl.FP32)
                init2_0 = pl.create_tensor([64], dtype=pl.FP32)
                for i_0, (a_0, b_0) in pl.range(5, init_values=(init1_0, init2_0)):
                    if i_0 == 0:
                        new_a_0 = pl.mul(a_0, 2.0)
                        new_b_0 = pl.mul(b_0, 3.0)
                        out_a_0, out_b_0 = pl.yield_(new_a_0, new_b_0)
                    else:
                        out_a_0, out_b_0 = pl.yield_(a_0, b_0)
                    res_a_0, res_b_0 = pl.yield_(out_a_0, out_b_0)
                result_0 = pl.add(res_a_0, res_b_0)
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
                tmp_0_0 = pl.create_tensor([64], dtype=pl.FP32)
                tmp_1_0 = pl.add(tmp_0_0, x)
                result_0 = pl.add(tmp_1_0, tmp_0_0)
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
                result_0 = pl.add(x, y)
                result_1 = pl.add(result_0, z)
                return result_1

        After = passes.convert_to_ssa()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_tensor_view_valid_shape_substitution(self):
        """Variables in TensorView.valid_shape must be renamed after SSA (issue #853)."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                x: pl.Tensor[[16, 64], pl.FP32],
                n: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[8, 64], pl.FP32]:
                valid_len: pl.Scalar[pl.INDEX] = pl.min(n, 64)
                chunk: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.slice(
                    x, [8, 64], [0, 0], valid_shape=[8, valid_len]
                )
                return chunk

        @pl.program
        class Expected:
            @pl.function(strict_ssa=True)
            def main(
                self,
                x: pl.Tensor[[16, 64], pl.FP32],
                n: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[8, 64], pl.FP32]:
                valid_len_0 = pl.min(n, 64)
                chunk_0 = pl.tensor.slice(x, [8, 64], [0, 0], valid_shape=[8, valid_len_0])
                return chunk_0

        After = passes.convert_to_ssa()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_tile_view_valid_shape_substitution(self):
        """Variables in TileType.tile_view.valid_shape must be renamed after SSA.

        Dual of test_tensor_view_valid_shape_substitution: the pass walks the
        Call's return type and rewrites the TileView.valid_shape branch
        (convert_to_ssa_pass.cpp SubstType, TileType case). A scalar that is
        SSA-versioned must propagate into the resulting Tile's valid_shape.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                x: pl.Tensor[[16, 64], pl.FP32],
                n: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[8, 64], pl.FP32]:
                valid_len: pl.Scalar[pl.INDEX] = pl.min(n, 64)
                t = pl.load(x, [0, 0], [16, 64])
                s = pl.tile.slice(t, [8, 64], [0, 0], valid_shape=[8, valid_len])
                out = pl.store(s, [0, 0], x)
                return out

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore, strict_ssa=True)
            def main(
                self,
                x: pl.Tensor[[16, 64], pl.FP32],
                n: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[8, 64], pl.FP32]:
                valid_len_0: pl.Scalar[pl.INDEX] = pl.min(n, 64)
                t_0 = pl.load(x, [0, 0], [16, 64])
                s_0 = pl.tile.slice(t_0, [8, 64], [0, 0], valid_shape=[8, valid_len_0])
                out_0 = pl.store(s_0, [0, 0], x)
                return out_0

        After = passes.convert_to_ssa()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_distributed_tensor_view_valid_shape_substitution(self):
        """``SubstType`` must preserve ``DistributedTensorType`` (and its
        ``window_buffer_`` back-reference) when shape / view substitution
        runs on a distributed tensor type — otherwise the SSA-versioned Var
        silently downgrades to a plain ``TensorType`` and downstream passes
        (``CollectCommGroups``, codegen) lose the comm-group binding.

        The dynamic-shape ``valid_shape`` is what forces the
        ``changed=true`` branch in ``SubstType``; the static-shape happy
        path is covered by the polymorphism tests in
        ``tests/ut/language/parser/test_distributed_tensor_polymorphism.py``.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                data: pl.InOut[pld.DistributedTensor[[16, 64], pl.FP32]],
                n: pl.Scalar[pl.INDEX],
                out: pl.Out[pl.Tensor[[8, 64], pl.FP32]],
            ) -> pl.Tensor[[8, 64], pl.FP32]:
                valid_len: pl.Scalar[pl.INDEX] = pl.min(n, 64)
                sub = pl.tensor.slice(data, [8, 64], [0, 0], valid_shape=[8, valid_len])
                tile = pl.load(sub, [0, 0], [8, 64])
                return pl.store(tile, [0, 0], out)

        After = passes.convert_to_ssa()(Before)

        # Locate the post-SSA tensor.slice Call and assert its return type
        # remains a DistributedTensorType (window_buffer_ stays None — it is
        # populated later by CollectCommGroups; the load-bearing invariant
        # here is the preserved ObjectKind).
        gvar = After.get_global_var("main")
        assert gvar is not None
        func = After.functions[gvar]

        slice_calls: list[ir.Call] = []

        def walk(stmt: ir.Stmt) -> None:
            if isinstance(stmt, ir.AssignStmt) and isinstance(stmt.value, ir.Call):
                if stmt.value.op.name == "tensor.slice":
                    slice_calls.append(stmt.value)
            if isinstance(stmt, ir.SeqStmts):
                for s in stmt.stmts:
                    walk(s)
            if isinstance(stmt, ir.ForStmt):
                walk(stmt.body)
            if isinstance(stmt, ir.IfStmt):
                walk(stmt.then_body)
                if stmt.else_body is not None:
                    walk(stmt.else_body)

        walk(func.body)
        assert len(slice_calls) == 1
        t = slice_calls[0].type
        assert isinstance(t, ir.DistributedTensorType), (
            f"SSA substitution must preserve DistributedTensorType, got {type(t).__name__}"
        )

    def test_unused_variable(self):
        """Unused variable should still be versioned."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                _unused: pl.Tensor[[64], pl.FP32] = pl.mul(x, 3.0)
                result: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
                return result

        @pl.program
        class Expected:
            @pl.function(strict_ssa=True)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                _unused_0 = pl.mul(x, 3.0)
                result_0 = pl.add(x, 1.0)
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
                acc_0 = pl.create_tensor([64], dtype=pl.FP32)
                for i_0, (acc_iter_1,) in pl.range(0, 10, 1, init_values=(acc_0,)):
                    acc_2 = pl.add(acc_iter_1, 1.0)
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
                result_0 = x_0
                for i_0, (result_iter_1,) in pl.range(0, 5, 1, init_values=(result_0,)):
                    result_2 = pl.add(result_iter_1, 1.0)
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
                a_0 = x_0
                b_0 = pl.mul(x_0, 2.0)
                for i_0, (a_iter_1, b_iter_2) in pl.range(0, 3, 1, init_values=(a_0, b_0)):
                    a_3 = pl.add(a_iter_1, 1.0)
                    b_4 = pl.mul(b_iter_2, 1.5)
                    a_1, b_2 = pl.yield_(a_3, b_4)
                result_0 = pl.add(a_1, b_2)
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
                    temp_0 = pl.mul(x_0, 2.0)
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
                result_0 = x_0
                for i_0, (result_iter_1,) in pl.range(0, 3, 1, init_values=(result_0,)):
                    for j_0, (result_iter_2,) in pl.range(0, 2, 1, init_values=(result_iter_1,)):
                        result_3 = pl.add(result_iter_2, 1.0)
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
                outer_0 = x_0
                inner_0 = pl.mul(x_0, 2.0)
                for i_0, (inner_iter_1, outer_iter_1) in pl.range(0, 2, 1, init_values=(inner_0, outer_0)):
                    for j_0, (inner_iter_3,) in pl.range(0, 3, 1, init_values=(inner_iter_1,)):
                        inner_5 = pl.add(inner_iter_3, 1.0)
                        inner_4 = pl.yield_(inner_5)
                    outer_3 = pl.add(outer_iter_1, inner_4)
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
                result_0 = x_0
                for i_0, (result_iter_1,) in pl.range(0, 5, 1, init_values=(result_0,)):
                    if i_0 == 0:
                        result_3 = pl.mul(result_iter_1, 2.0)
                        result_5 = pl.yield_(result_3)
                    else:
                        result_4 = pl.add(result_iter_1, 1.0)
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
                result_0 = x_0
                for i_0, (result_iter_1,) in pl.range(0, 3, 1, init_values=(result_0,)):
                    for j_0, (result_iter_3,) in pl.range(0, 2, 1, init_values=(result_iter_1,)):
                        if j_0 == 0:
                            result_5 = pl.add(result_iter_3, 1.0)
                            result_7 = pl.yield_(result_5)
                        else:
                            result_6 = pl.mul(result_iter_3, 1.5)
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
                a_0 = x_0
                b_0 = pl.mul(x_0, 2.0)
                for i_0, (a_iter_1, b_iter_1) in pl.range(0, 2, 1, init_values=(a_0, b_0)):
                    if i_0 == 0:
                        for j_0, (a_iter_3,) in pl.range(0, 2, 1, init_values=(a_iter_1,)):
                            a_5 = pl.add(a_iter_3, 1.0)
                            a_4 = pl.yield_(a_5)
                        a_6, b_4 = pl.yield_(a_4, b_iter_1)
                    else:
                        b_3 = pl.mul(b_iter_1, 2.0)
                        a_6, b_4 = pl.yield_(a_iter_1, b_3)
                    a_2, b_2 = pl.yield_(a_6, b_4)
                result_0 = pl.add(a_2, b_2)
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
                result_0 = x_0
                for i_0, (result_iter_1,) in pl.range(0, 2, 1, init_values=(result_0,)):
                    result_2 = pl.add(result_iter_1, 1.0)
                    result_1 = pl.yield_(result_2)
                for j_0, (result_iter_3,) in pl.range(0, 3, 1, init_values=(result_1,)):
                    result_4 = pl.mul(result_iter_3, 1.5)
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
                a_0 = x_0
                b_0 = pl.mul(x_0, 2.0)
                for i_0, (a_iter_1, b_iter_1) in pl.range(0, 1, 1, init_values=(a_0, b_0)):
                    if i_0 == 0:
                        a_3 = pl.add(a_iter_1, 1.0)
                        a_4, b_4 = pl.yield_(a_3, b_iter_1)
                    else:
                        b_3 = pl.add(b_iter_1, 1.0)
                        a_4, b_4 = pl.yield_(a_iter_1, b_3)
                    a_2, b_2 = pl.yield_(a_4, b_4)
                result_0 = pl.add(a_2, b_2)
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
                result_0 = x_0
                for i_0, (result_iter_1,) in pl.range(0, 3, 1, init_values=(result_0,)):
                    result_2 = pl.add(result_iter_1, 1.0)
                    result_1 = pl.yield_(result_2)
                final_0 = pl.mul(result_1, 2.0)
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
                result_0 = x_0
                for i_0, (result_iter_1,) in pl.range(0, 10, 1, init_values=(result_0,)):
                    if i_0 > 5:
                        result_4 = pl.yield_(result_iter_1)
                    else:
                        result_3 = pl.add(result_iter_1, 1.0)
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
        add_call = ir.Call(add_op, [a, c], tensor_type, span)
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
        """Variable first assigned inside for loop, used after loop.

        The escaping variable ``out`` is promoted to an iter_arg of the loop
        (init_values = the Out param ``c``) and the loop's return var (``out_rv``)
        carries the escaped value out to the post-loop return.
        """
        program = self._build_for_loop_escaping_program()

        @pl.program
        class Expected:
            @pl.function(strict_ssa=True)
            def main(
                self,
                a: pl.Tensor[[64], pl.FP32],
                c: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                for i_0, (out_iter_0,) in pl.range(0, 4, 1, init_values=(c,)):
                    out_2 = pl.add(a, c)
                    out_1 = pl.yield_(out_2)
                return out_1

        After = passes.convert_to_ssa()(program)
        ir.assert_structural_equal(After, Expected)

    def test_for_loop_non_escaping_var_not_promoted(self):
        """Variable defined inside loop but NOT used after should not be promoted.

        ``tmp`` is a pure loop-local: it stays a plain assignment inside the
        body and never appears in ``init_values`` or ``yield_``. Only ``result``
        is loop-carried.
        """

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                result: pl.Tensor[[64], pl.FP32] = x
                for i in pl.range(4):
                    tmp: pl.Tensor[[64], pl.FP32] = pl.add(result, 1.0)
                    result = pl.add(tmp, 1.0)
                return result

        @pl.program
        class Expected:
            @pl.function(strict_ssa=True)
            def main(self, x_0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                result_0 = x_0
                for i_0, (result_iter_1,) in pl.range(0, 4, 1, init_values=(result_0,)):
                    tmp_0 = pl.add(result_iter_1, 1.0)
                    result_2 = pl.add(tmp_0, 1.0)
                    result_1 = pl.yield_(result_2)
                return result_1

        After = passes.convert_to_ssa()(Before)
        ir.assert_structural_equal(After, Expected)

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
                c_out = self.add_kernel(a, b, c)
                return c_out

        After = passes.convert_to_ssa()(Before)
        # The input is already single-assignment, so SSA conversion is a no-op:
        # the structure (including dynamic-shape params M, N) is unchanged.
        ir.assert_structural_equal(After, Before)
        # The original intent of this test is that dynamic shape vars do not
        # trigger a scope violation — keep the explicit verifier check.
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
                a_0 = x_0
                b_0 = pl.mul(x_0, 2.0)
                for i_0, (a_iter_1,) in pl.range(0, 4, 1, init_values=(a_0,)):
                    tmp_0 = pl.add(x_0, 1.0)
                    a_2 = pl.add(a_iter_1, tmp_0)
                    a_1 = pl.yield_(a_2)
                for j_0, (b_iter_1,) in pl.range(0, 4, 1, init_values=(b_0,)):
                    tmp_1 = pl.add(x_0, 2.0)
                    b_2 = pl.add(b_iter_1, tmp_1)
                    b_1 = pl.yield_(b_2)
                result_0 = pl.add(a_1, b_1)
                return result_0

        After = passes.convert_to_ssa()(Before)
        ir.assert_structural_equal(After, Expected)

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
        add_call = ir.Call(add_op, [x, c], tensor_type, span)
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

        # Expected: `out` is pre-registered as an iter_arg (init = Out param `c`).
        # The IfStmt produces a phi: the empty then-branch yields the pass-through
        # iter value (out_iter), the else-branch yields the newly computed value.
        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore, strict_ssa=True)
            def kernel(
                self,
                x: pl.Tensor[[64], pl.FP32],
                c: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                for i_0, (out_iter_0,) in pl.range(0, 4, 1, init_values=(c,)):
                    if i_0 % 2 != 0:
                        out_phi_0 = pl.yield_(out_iter_0)
                    else:
                        out_2 = pl.add(x, c)
                        out_phi_0 = pl.yield_(out_2)
                    out_1 = pl.yield_(out_phi_0)
                return out_1

        After = passes.convert_to_ssa()(program)
        ir.assert_structural_equal(After, Expected)

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

        # ``k0`` and ``chunk`` are recomputed each iteration and stay plain
        # assignments — only ``acc`` is loop-carried (init_values + yield_).
        @pl.program
        class Expected:
            @pl.function(strict_ssa=True)
            def main(self, x_0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                acc_0 = x_0
                for i_0, (acc_iter_1,) in pl.range(0, 4, 1, init_values=(acc_0,)):
                    k0_0 = i_0 * 8
                    chunk_0 = pl.add(x_0, k0_0)
                    acc_2 = pl.add(acc_iter_1, chunk_0)
                    acc_1 = pl.yield_(acc_2)
                result_0 = pl.mul(acc_1, 2.0)
                return result_0

        After = passes.convert_to_ssa()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_loop_local_var_not_carried_as_iter_arg(self):
        """Loop-local variable freshly created each iteration must not be loop-carried.

        Regression test for issue #642: ConvertToSSA incorrectly promoted
        a variable created inside a loop body to a loop-carry iter_arg.
        """

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[16, 128], pl.FP32]) -> pl.Tensor[[16, 128], pl.FP32]:
                result = pl.create_tensor([16, 1], dtype=pl.FP32)
                result = pl.mul(result, 0.0)
                for b0 in pl.range(0, 16, 4):
                    result = pl.mul(result, 2.0)
                out = pl.mul(result, 2.0)

                # local_var is freshly created each iteration — must NOT
                # become a loop-carry variable in the outer loop.
                for b0 in pl.range(0, 16, 4):
                    local_var = pl.create_tensor([4, 1], dtype=pl.FP32)
                    local_var = pl.mul(local_var, 0.0)
                    for kb in pl.range(0, 40):
                        local_var = pl.add(local_var, 1.0)
                    _tmp = pl.mul(local_var, 3.0)

                return out

        # Only the first ``b0`` loop and the inner ``kb`` loop carry a variable.
        # The second ``b0`` loop has no init_values: ``local_var`` is freshly
        # created each iteration and never escapes, so it is not loop-carried.
        @pl.program
        class Expected:
            @pl.function(strict_ssa=True)
            def main(self, x_0: pl.Tensor[[16, 128], pl.FP32]) -> pl.Tensor[[16, 128], pl.FP32]:
                result_0 = pl.create_tensor([16, 1], dtype=pl.FP32)
                result_1 = pl.mul(result_0, 0.0)
                for b0_0, (result_iter_2,) in pl.range(0, 16, 4, init_values=(result_1,)):
                    result_4 = pl.mul(result_iter_2, 2.0)
                    result_3 = pl.yield_(result_4)
                out_0 = pl.mul(result_3, 2.0)
                for b0_1 in pl.range(0, 16, 4):
                    local_var_0 = pl.create_tensor([4, 1], dtype=pl.FP32)
                    local_var_1 = pl.mul(local_var_0, 0.0)
                    for kb_0, (local_var_iter_2,) in pl.range(0, 40, 1, init_values=(local_var_1,)):
                        local_var_4 = pl.add(local_var_iter_2, 1.0)
                        local_var_3 = pl.yield_(local_var_4)
                    _tmp_0 = pl.mul(local_var_3, 3.0)
                return out_0

        After = passes.convert_to_ssa()(Before)
        ir.assert_structural_equal(After, Expected)


class TestSubmitSSA:
    """SSA renaming must reach into Submit's first-class ``args_`` and ``deps_``.

    ``Submit`` is a sibling call-like Expr kind (see pass-submit-awareness.md):
    ConvertToSSA's ExprSubstituter overrides VisitExpr_(SubmitPtr) and relies on
    the base IRMutator to walk both ``args_`` and ``deps_``. A reassigned arg
    must rebind to its latest SSA version in ``args_``, and a producer TaskId
    used in a consumer's ``deps=[...]`` must rebind to the versioned TaskId Var.
    """

    def test_submit_args_and_deps_versioned(self):
        """A reassigned Submit arg picks up its latest SSA version, and a
        downstream ``deps=[a_tid]`` rebinds to the versioned producer TaskId."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def producer(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.InCore)
            def consumer(
                self,
                x: pl.Tensor[[64], pl.FP32],
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                x: pl.Tensor[[64], pl.FP32],
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                # Reassign x so SSA mints a fresh version; the producer Submit
                # must reference that latest version in args_, not the param.
                x = pl.add(x, 1.0)
                with pl.manual_scope():
                    a, a_tid = pl.submit(self.producer, x)
                    # deps=[a_tid] lives on Submit::deps_ and must be versioned too.
                    b, b_tid = pl.submit(self.consumer, a, out, deps=[a_tid])
                return out

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore, strict_ssa=True)
            def producer(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.InCore, strict_ssa=True)
            def consumer(
                self,
                x: pl.Tensor[[64], pl.FP32],
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                return out

            @pl.function(type=pl.FunctionType.Orchestration, strict_ssa=True)
            def main(
                self,
                x: pl.Tensor[[64], pl.FP32],
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                x_1 = pl.add(x, 1.0)
                with pl.manual_scope():
                    a, a_tid = pl.submit(self.producer, x_1)
                    b, b_tid = pl.submit(self.consumer, a, out, deps=[a_tid])
                return out

        After = passes.convert_to_ssa()(Before)
        ir.assert_structural_equal(After, Expected)


class TestMidBodyYieldGuard:
    """ConvertToSSA rejects a body whose SeqStmts contains a YieldStmt before
    the trailing position via INTERNAL_CHECK in ExtractYield/ReplaceOrAppendYield.

    Structural pre-verification (NoRedundantBlocks) normally intercepts this
    shape before ConvertToSSA runs, so the test wraps the call in a
    VerificationLevel.NONE PassContext to exercise the in-pass assertion.
    """

    def test_for_body_with_mid_body_yield_rejected(self):
        """ForStmt body shaped as [YieldStmt, AssignStmt] trips
        AssertNoMidBodyYield. Constructed directly because the DSL parser
        cannot produce this shape; iter_args ensure the loop-handler path
        that calls the helpers is exercised.
        """
        span = ir.Span.unknown()

        a = ir.Var("a", ir.ScalarType(DataType.INT64), span)
        params: list[ir.Var] = [a]
        return_types: list[ir.Type] = [ir.ScalarType(DataType.INT64)]

        loop_var = ir.Var("i", ir.ScalarType(DataType.INDEX), span)
        iter_arg = ir.IterArg("acc", ir.ScalarType(DataType.INT64), a, span)

        mid_yield = ir.YieldStmt([iter_arg], span)
        trailing_assign = ir.AssignStmt(ir.Var("dummy", ir.ScalarType(DataType.INT64), span), loop_var, span)
        body = ir.SeqStmts([mid_yield, trailing_assign], span)

        rv = ir.Var("result", ir.ScalarType(DataType.INT64), span)
        for_stmt = ir.ForStmt(
            loop_var,
            ir.ConstInt(0, DataType.INDEX, span),
            ir.ConstInt(10, DataType.INDEX, span),
            ir.ConstInt(1, DataType.INDEX, span),
            [iter_arg],
            body,
            [rv],
            span,
        )

        func_body = ir.SeqStmts([for_stmt, ir.ReturnStmt([rv], span)], span)
        func = ir.Function("main", params, return_types, func_body, span)
        program = ir.Program([func], "test_program", span)

        ctx = passes.PassContext([], passes.VerificationLevel.NONE)
        with ctx, pytest.raises(Exception, match="YieldStmt at position"):
            passes.convert_to_ssa()(program)

    def test_function_body_with_mid_body_yield_rejected(self):
        """Function-body SeqStmts with a mid-body YieldStmt trips
        AssertNoMidBodyYield via ConvertSeq — the path that bypasses the
        loop/if scope handlers.
        """
        span = ir.Span.unknown()

        a = ir.Var("a", ir.ScalarType(DataType.INT64), span)
        params: list[ir.Var] = [a]
        return_types: list[ir.Type] = [ir.ScalarType(DataType.INT64)]

        dummy_var = ir.Var("dummy", ir.ScalarType(DataType.INT64), span)
        assign = ir.AssignStmt(dummy_var, a, span)
        mid_yield = ir.YieldStmt([a], span)
        ret = ir.ReturnStmt([a], span)
        func_body = ir.SeqStmts([assign, mid_yield, ret], span)
        func = ir.Function("main", params, return_types, func_body, span)
        program = ir.Program([func], "test_program", span)

        ctx = passes.PassContext([], passes.VerificationLevel.NONE)
        with ctx, pytest.raises(Exception, match="YieldStmt at position"):
            passes.convert_to_ssa()(program)


class TestCallAttrSubstitution:
    """SSA renaming must reach into Call attrs that hold IR expressions.

    Regression: ``Call.attrs["device"]`` (an ``ExprPtr`` written by the N3
    parser on host_orch → chip_orch dispatches) used to be left untouched by
    ConvertToSSA, leaving a dead reference to the pre-SSA loop induction var.
    CollectCommGroups then failed identity-matching the dead Var against any
    enclosing ForStmt's versioned ``loop_var_``.
    """

    @staticmethod
    def _find_host_orch_dispatch(program: ir.Program) -> tuple[ir.ForStmt, ir.Call]:
        """Locate the host_orch ForStmt and the inner chip_orch dispatch Call."""

        host_orch = None
        for _gv, func in program.functions.items():
            if func.name == "host_orch":
                host_orch = func
                break
        assert host_orch is not None, "host_orch function not found in program"

        for_stmt: ir.ForStmt | None = None
        call: ir.Call | None = None

        def walk_stmt(stmt: ir.Stmt) -> None:
            nonlocal for_stmt, call
            if isinstance(stmt, ir.ForStmt):
                for_stmt = stmt
                walk_stmt(stmt.body)
                return
            if isinstance(stmt, ir.SeqStmts):
                for s in stmt.stmts:
                    walk_stmt(s)
                return
            if isinstance(stmt, ir.EvalStmt):
                if isinstance(stmt.expr, ir.Call) and "device" in dict(stmt.expr.attrs):
                    call = stmt.expr
                return
            if isinstance(stmt, ir.AssignStmt):
                if isinstance(stmt.value, ir.Call) and "device" in dict(stmt.value.attrs):
                    call = stmt.value
                return

        walk_stmt(host_orch.body)
        assert for_stmt is not None, "expected a ForStmt in host_orch"
        assert call is not None, "expected a chip_orch dispatch Call with device= attr"
        return for_stmt, call

    def test_device_var_attr_versioned_with_loop_var(self):
        """``device=r`` must rebind to the versioned induction Var, not the dead
        pre-SSA Var. Verified by object-identity between the ForStmt's
        ``loop_var`` and the Call's ``attrs['device']`` after SSA."""
        SIZE = 8

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Orchestration)
            def chip_orch(self, out: pl.Out[pl.Tensor[[SIZE], pl.FP32]]) -> pl.Tensor[[SIZE], pl.FP32]:
                return out

            @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
            def host_orch(
                self,
                outputs: pl.Out[pl.Tensor[[2, SIZE], pl.FP32]],
            ) -> pl.Tensor[[2, SIZE], pl.FP32]:
                for r in pl.range(2):
                    self.chip_orch(outputs[r], device=r)
                return outputs

        After = passes.convert_to_ssa()(Before)
        for_stmt, call = self._find_host_orch_dispatch(After)
        device_expr = dict(call.attrs)["device"]
        assert isinstance(device_expr, ir.Var), (
            f"device= attr should remain an ir.Var after SSA, got {type(device_expr).__name__}"
        )
        assert device_expr is for_stmt.loop_var, (
            "ConvertToSSA must substitute the device= attr Var to point at the "
            f"versioned loop_var ({for_stmt.loop_var.name_hint!r}); got "
            f"{device_expr.name_hint!r}"
        )

    def test_device_var_attr_versioned_before_after(self):
        """Full structural before/after for the ``device=r`` Call-attr rewrite.

        Complements the object-identity check above: the whole host_orch loop
        must round-trip with ``device=`` rebound to the versioned induction Var
        (``r_0``) — the same Var the slice ``outputs[r_0]`` uses. The pass
        rewrites the attr ExprPtr in ConvertScope/SubstCallAttrs (kAttrDevice).
        """
        SIZE = 8

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Orchestration)
            def chip_orch(self, out: pl.Out[pl.Tensor[[SIZE], pl.FP32]]) -> pl.Tensor[[SIZE], pl.FP32]:
                return out

            @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
            def host_orch(
                self,
                outputs: pl.Out[pl.Tensor[[2, SIZE], pl.FP32]],
            ) -> pl.Tensor[[2, SIZE], pl.FP32]:
                for r in pl.range(2):
                    self.chip_orch(outputs[r], device=r)
                return outputs

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.Orchestration, strict_ssa=True)
            def chip_orch(self, out: pl.Out[pl.Tensor[[SIZE], pl.FP32]]) -> pl.Tensor[[SIZE], pl.FP32]:
                return out

            @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator, strict_ssa=True)
            def host_orch(
                self,
                outputs: pl.Out[pl.Tensor[[2, SIZE], pl.FP32]],
            ) -> pl.Tensor[[2, SIZE], pl.FP32]:
                for r_0 in pl.range(0, 2, 1):
                    self.chip_orch(outputs[r_0], device=r_0)
                return outputs

        After = passes.convert_to_ssa()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_device_const_attr_preserved(self):
        """``device=<ConstInt>`` is untouched by SSA — it has no Var to rewrite,
        so the Call's attrs vector should not be rebuilt unnecessarily."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Orchestration)
            def chip_orch(self, out: pl.Out[pl.Tensor[[8], pl.FP32]]) -> pl.Tensor[[8], pl.FP32]:
                return out

            @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
            def host_orch(
                self,
                outputs: pl.Out[pl.Tensor[[2, 8], pl.FP32]],
            ) -> pl.Tensor[[2, 8], pl.FP32]:
                self.chip_orch(outputs[0], device=0)
                self.chip_orch(outputs[1], device=1)
                return outputs

        After = passes.convert_to_ssa()(Before)
        host_orch = next(f for _, f in After.functions.items() if f.name == "host_orch")

        seen: list[int] = []

        def walk(stmt: ir.Stmt) -> None:
            if isinstance(stmt, ir.SeqStmts):
                for s in stmt.stmts:
                    walk(s)
            elif isinstance(stmt, ir.EvalStmt):
                v = stmt.expr
                if isinstance(v, ir.Call) and "device" in dict(v.attrs):
                    dev = dict(v.attrs)["device"]
                    assert isinstance(dev, ir.ConstInt)
                    seen.append(dev.value)
            elif isinstance(stmt, ir.AssignStmt):
                v = stmt.value
                if isinstance(v, ir.Call) and "device" in dict(v.attrs):
                    dev = dict(v.attrs)["device"]
                    assert isinstance(dev, ir.ConstInt)
                    seen.append(dev.value)

        walk(host_orch.body)
        assert seen == [0, 1], f"expected device=[0, 1] preserved, got {seen}"


class TestSpmdCoreNumSubstitution:
    """SSA renaming must reach into ``SpmdScopeStmt.core_num_``.

    Regression for issue #1550: ``core_num_`` is a direct ``ExprPtr`` field
    on ``SpmdScopeStmt`` (not in ``attrs``). ConvertToSSA's ``ConvertScope``
    rewrote ``body_`` and ``attrs_`` but never touched ``core_num_``, so a
    Var defined in the enclosing scope was left at its pre-SSA pointer —
    the printer marked it ``__FREE_VAR`` and the SSA verifier rejected the
    IR with ``Variable 'n_rows' used outside its defining scope``.
    """

    @staticmethod
    def _walk(stmt: ir.Stmt) -> list[ir.Stmt]:
        """DFS pre-order over all stmts reachable from ``stmt``."""
        out: list[ir.Stmt] = [stmt]
        if isinstance(stmt, ir.SeqStmts):
            for s in stmt.stmts:
                out.extend(TestSpmdCoreNumSubstitution._walk(s))
        elif isinstance(stmt, ir.ForStmt):
            out.extend(TestSpmdCoreNumSubstitution._walk(stmt.body))
        elif isinstance(stmt, ir.ScopeStmt):
            out.extend(TestSpmdCoreNumSubstitution._walk(stmt.body))
        elif isinstance(stmt, ir.IfStmt):
            out.extend(TestSpmdCoreNumSubstitution._walk(stmt.then_body))
            if stmt.else_body is not None:
                out.extend(TestSpmdCoreNumSubstitution._walk(stmt.else_body))
        return out

    @staticmethod
    def _collect_var_unique_ids(expr: ir.Expr, name_prefix: str) -> list[tuple[str, int]]:
        """Return ``(name_hint, unique_id)`` for every Var under ``expr`` whose
        name_hint starts with ``name_prefix``. ``ir.Var`` is not hashable, so we
        use the stable ``unique_id`` for identity comparisons."""
        seen: list[tuple[str, int]] = []

        class _Collector(ir.IRVisitor):
            def visit_var(self, op: ir.Var) -> None:
                if op.name_hint.startswith(name_prefix):
                    seen.append((op.name_hint, op.unique_id))

        _Collector().visit_expr(expr)
        return seen

    def test_spmd_core_num_substituted_to_outer_scope_var(self):
        """A scalar bound in a ``pl.parallel`` body must SSA-rename inside
        the nested ``pl.spmd(core_num=<expr>)`` argument, not just inside
        the spmd body. Without the fix the SSA verifier raises
        ``Variable 'n_rows' used outside its defining scope``."""
        N, T_TILE, T_MAX, SUB_BLOCKS = 8, 32, 192, 4

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Orchestration)
            def repro(
                self,
                counts: pl.Tensor[[N, 1], pl.INT32],
                out: pl.Out[pl.Tensor[[N, T_MAX], pl.FP32]],
            ) -> pl.Tensor[[N, T_MAX], pl.FP32]:
                for i in pl.parallel(N):
                    n_rows = pl.read(counts, [i, 0])
                    # n_rows appears in the spmd core_num arg — the broken
                    # position — and is reused inside the body. Before the fix
                    # only the body reference is renamed; the printer marks
                    # the core_num occurrence ``__FREE_VAR``.
                    for s in pl.spmd(
                        ((n_rows + T_TILE - 1) // T_TILE) * SUB_BLOCKS,  # pyright: ignore[reportArgumentType]
                        name_hint="dyn",
                    ):
                        t = s // SUB_BLOCKS
                        t0 = t * T_TILE
                        offset = pl.min(t0, n_rows)
                        tile = pl.full([T_TILE, T_TILE], dtype=pl.FP32, value=1.0)
                        out[i : i + 1, offset : offset + T_TILE] = pl.reshape(pl.row_sum(tile), [1, T_TILE])
                return out

        After = passes.convert_to_ssa()(Before)
        # The SSA verifier raises if core_num still references the pre-SSA Var
        # (which the printer prints as ``__FREE_VAR``). Restrict the verifier
        # to SSAForm so the test stays focused on the regression — the
        # default property set includes unrelated checks (NoNestedCall, ...)
        # that the unsimplified Before IR doesn't satisfy.
        ssa_only = passes.IRPropertySet()
        ssa_only.insert(passes.IRProperty.SSAForm)
        passes.run_verifier(ssa_only)(After)

        # Identity check: the n_rows Var referenced inside core_num must be
        # the same object as the LHS of the AssignStmt ``n_rows = pl.read(...)``
        # in the enclosing pl.parallel body — not a stale pre-SSA pointer.
        fn = next(f for _, f in After.functions.items() if f.name == "repro")
        stmts = self._walk(fn.body)
        n_rows_assign = next(
            s for s in stmts if isinstance(s, ir.AssignStmt) and s.var.name_hint.startswith("n_rows")
        )
        spmd = next(s for s in stmts if isinstance(s, ir.SpmdScopeStmt))

        core_num_n_rows = self._collect_var_unique_ids(spmd.core_num, "n_rows")
        assigned = (n_rows_assign.var.name_hint, n_rows_assign.var.unique_id)
        assert core_num_n_rows and all(entry == assigned for entry in core_num_n_rows), (
            f"SpmdScopeStmt.core_num must reference the SSA-versioned n_rows Var "
            f"({assigned}); got {core_num_n_rows}"
        )


class TestScopeTransparentToSSA:
    """RuntimeScopeStmt is transparent to SSA: a for/if body wrapped in a
    ``with pl.scope()`` keeps its carry-yield *inside* the scope, and
    ConvertToSSA / SSAVerify must look through the scope to associate it."""

    @staticmethod
    def _verify_ssa(program):
        ps = passes.IRPropertySet()
        ps.insert(passes.IRProperty.SSAForm)
        passes.run_verifier(ps)(program)  # raises on violation

    def test_loop_carried_yield_inside_scope_converts_and_verifies(self):
        from pypto import backend  # noqa: PLC0415
        from pypto.backend import BackendType  # noqa: PLC0415

        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)
        try:

            @pl.program
            class Prog:
                @pl.function(type=pl.FunctionType.AIV)
                def kernel(
                    self,
                    a: pl.Tensor[[16, 16], pl.FP32],
                    out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
                ) -> pl.Tensor[[16, 16], pl.FP32]:
                    t: pl.Tile[[16, 16], pl.FP32] = pl.load(a, [0, 0], [16, 16])
                    r: pl.Tensor[[16, 16], pl.FP32] = pl.store(t, [0, 0], out)
                    return r

                @pl.function(type=pl.FunctionType.Orchestration, auto_scope=False)
                def orch(self, a: pl.Tensor[[16, 16], pl.FP32], out: pl.Out[pl.Tensor[[16, 16], pl.FP32]]):
                    for i, (acc,) in pl.range(4, init_values=(out,)):
                        with pl.scope():
                            nxt: pl.Tensor[[16, 16], pl.FP32] = self.kernel(a, acc)
                            acc = pl.yield_(nxt)
                    return acc

            after = passes.convert_to_ssa()(Prog)
            # Scope is transparent → SSAForm holds (carry-yield seen through scope).
            self._verify_ssa(after)
        finally:
            backend.reset_for_testing()


class TestScopeOutlineBoundary:
    """For non-``RuntimeScopeStmt`` scopes, ``ConvertScope`` blocks
    inner-loop escaping-var promotion for variables first-defined inside
    the scope body. ``cur_`` stays transparent (later passes such as
    ``InterchangeChunkLoops`` rely on scope-local vars flowing out for
    sequential references), but the escaping path is gated to avoid the
    ``init_values=(foo__FREE_VAR,)`` failure mode. Regression for #1351."""

    @staticmethod
    def _collect_for_stmts(stmt):
        out: list[ir.ForStmt] = []

        def walk(s):
            if s is None:
                return
            if isinstance(s, ir.ForStmt):
                out.append(s)
            for attr in ("body", "then_body", "else_body"):
                child = getattr(s, attr, None)
                if child is not None:
                    walk(child)
            if isinstance(s, ir.SeqStmts):
                for child in s.stmts:
                    walk(child)

        walk(stmt)
        return out

    def test_scope_blocks_escaping_var_promotion_in_inner_loop(self):
        """Issue #1351. A var first-defined inside ``pl.at`` body must NOT
        be promoted to inner-loop ``init_values`` just because some
        subsequent statement (outside the scope) references it.

        Pre-fix produced ``init_values=(k0__FREE_VAR,)`` on both ``pl.parallel``
        and ``pl.range`` (FindInitValue had no pre-loop scalar of the right
        type, so it created an unversioned placeholder). The downstream
        ``k0__rv_*`` was then defined inside ``pl.at`` but used after it,
        which the SSA verifier rejected. Post-fix: neither inner loop
        carries ``k0``; the post-scope reference itself remains a user
        error caught by other verifiers."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                x: pl.Tensor[[256], pl.FP32],
                result: pl.Out[pl.Tensor[[256], pl.FP32]],
                out_scalar: pl.Out[pl.Scalar[pl.INDEX]],
            ) -> pl.Tensor[[256], pl.FP32]:
                K_CHUNK = 16
                with pl.at(level=pl.Level.CORE_GROUP):
                    for ob in pl.parallel(4):
                        for kb in pl.range(4):
                            k0 = kb * K_CHUNK
                            t = pl.load(x, [k0], [K_CHUNK])
                            pl.store(t, [k0], result)
                out_scalar = k0  # noqa: F841 — bug trigger: puts ``k0`` into future_needs
                return result

        # The post-scope ``out_scalar = k0`` is the user-side scope leak
        # that triggers the bug: it puts ``k0`` into ``future_needs`` at
        # the pl.at level. The leak itself is a user error (and downstream
        # property verifiers correctly flag it), so disable verification
        # here — the regression we are protecting is purely structural
        # (no bogus iter_args on the inner loops).
        with passes.PassContext([], passes.VerificationLevel.NONE):
            After = passes.convert_to_ssa()(Before)
        fn = next(f for _, f in After.functions.items() if f.name == "main")
        for_stmts = self._collect_for_stmts(fn.body)
        assert len(for_stmts) == 2, f"expected 2 for-loops, found {len(for_stmts)}"
        for loop in for_stmts:
            assert len(loop.iter_args) == 0, (
                f"loop with var {loop.loop_var.name_hint!r} got "
                f"iter_args={[ia.name_hint for ia in loop.iter_args]}; "
                f"scope-local ``k0`` must not be promoted past the pl.at boundary"
            )

    def test_scope_preserves_pre_existing_carried_var(self):
        """The scope-boundary trim must NOT block carried-var promotion of
        variables that exist *before* the scope (typical accumulator pattern).
        ``result`` here is an outer parameter that the inner loop updates;
        it must still be threaded through as an iter_arg/return_var."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                x: pl.Tensor[[64], pl.FP32],
                result: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    for i in pl.range(4):
                        result = pl.add(result, x)
                return result

        After = passes.convert_to_ssa()(Before)
        fn = next(f for _, f in After.functions.items() if f.name == "main")
        for_stmts = self._collect_for_stmts(fn.body)
        assert len(for_stmts) == 1
        loop = for_stmts[0]
        carried_names = [ia.name_hint for ia in loop.iter_args]
        # ``result`` is pre-existing → must be carried even inside pl.at.
        assert any(n.startswith("result") for n in carried_names), (
            f"pre-existing ``result`` must remain a carried iter_arg inside pl.at, "
            f"got iter_args={carried_names}"
        )

    def test_runtime_scope_remains_transparent_to_escaping(self):
        """``RuntimeScopeStmt`` (``pl.scope()``) is a thin codegen wrapper,
        NOT an outline boundary. A variable first-defined inside ``pl.scope()``
        and used after the scope must still be promoted through enclosing
        loops the same as in plain non-scoped code."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Orchestration, auto_scope=False)
            def main(
                self,
                x: pl.Tensor[[16, 16], pl.FP32],
                out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                for i, (acc,) in pl.range(4, init_values=(out,)):
                    with pl.scope():
                        nxt: pl.Tensor[[16, 16], pl.FP32] = pl.tensor.adds(acc, 1.0)
                        acc = pl.yield_(nxt)
                return acc

        After = passes.convert_to_ssa()(Before)
        ps = passes.IRPropertySet()
        ps.insert(passes.IRProperty.SSAForm)
        # Successful SSA verification is the key signal — pl.scope() must
        # stay transparent so the carry-yield inside it threads through.
        passes.run_verifier(ps)(After)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
