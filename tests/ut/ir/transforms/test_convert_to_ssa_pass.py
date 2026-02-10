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
from pypto import ir
from pypto.language.parser.diagnostics import SSAViolationError
from pypto.pypto_core import passes

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
                init = pl.create([64], dtype=pl.FP32)
                for i, (acc,) in pl.range(10, init_values=[init]):
                    new_acc = pl.add(acc, x)
                    result = pl.yield_(new_acc)
                return result

        @pl.program
        class Expected:
            @pl.function(strict_ssa=True)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                init_0: pl.Tensor[[64], pl.FP32] = pl.create([64], dtype=pl.FP32)
                for i_0, (acc_0,) in pl.range(10, init_values=[init_0]):
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
                init1: pl.Tensor[[64], pl.FP32] = pl.create([64], dtype=pl.FP32)
                init2: pl.Tensor[[64], pl.FP32] = pl.create([64], dtype=pl.FP32)
                for i, (acc1, acc2) in pl.range(5, init_values=[init1, init2]):
                    new1: pl.Tensor[[64], pl.FP32] = pl.add(acc1, x)
                    new2: pl.Tensor[[64], pl.FP32] = pl.mul(acc2, 2.0)
                    out1, out2 = pl.yield_(new1, new2)
                result: pl.Tensor[[64], pl.FP32] = pl.add(out1, out2)
                return result

        @pl.program
        class Expected:
            @pl.function(strict_ssa=True)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                init1_0: pl.Tensor[[64], pl.FP32] = pl.create([64], dtype=pl.FP32)
                init2_0: pl.Tensor[[64], pl.FP32] = pl.create([64], dtype=pl.FP32)
                for i_0, (acc1_0, acc2_0) in pl.range(5, init_values=[init1_0, init2_0]):
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
                init: pl.Tensor[[64], pl.FP32] = pl.create([64], dtype=pl.FP32)
                for i, (acc,) in pl.range(0, 10, 2, init_values=[init]):
                    new_acc: pl.Tensor[[64], pl.FP32] = pl.add(acc, x)
                    result = pl.yield_(new_acc)
                return result

        @pl.program
        class Expected:
            @pl.function(strict_ssa=True)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                init_0: pl.Tensor[[64], pl.FP32] = pl.create([64], dtype=pl.FP32)
                for i_0, (acc_0,) in pl.range(0, 10, 2, init_values=[init_0]):
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
                init: pl.Tensor[[64], pl.FP32] = pl.create([64], dtype=pl.FP32)
                for i, (outer,) in pl.range(3, init_values=[init]):
                    for j, (inner,) in pl.range(2, init_values=[outer]):
                        new_inner: pl.Tensor[[64], pl.FP32] = pl.add(inner, 1.0)
                        inner_out = pl.yield_(new_inner)
                    outer_out = pl.yield_(inner_out)
                return outer_out

        @pl.program
        class Expected:
            @pl.function(strict_ssa=True)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                init_0: pl.Tensor[[64], pl.FP32] = pl.create([64], dtype=pl.FP32)
                for i_0, (outer_0,) in pl.range(3, init_values=[init_0]):
                    for j_0, (inner_0,) in pl.range(2, init_values=[outer_0]):
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
                init: pl.Tensor[[64], pl.FP32] = pl.create([64], dtype=pl.FP32)
                for i, (acc,) in pl.range(5, init_values=[init]):
                    new_acc: pl.Tensor[[64], pl.FP32] = pl.add(acc, 1.0)
                    result1 = pl.yield_(new_acc)
                for j, (acc2,) in pl.range(3, init_values=[result1]):
                    new_acc2: pl.Tensor[[64], pl.FP32] = pl.mul(acc2, 2.0)
                    result2 = pl.yield_(new_acc2)
                return result2

        @pl.program
        class Expected:
            @pl.function(strict_ssa=True)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                init_0: pl.Tensor[[64], pl.FP32] = pl.create([64], dtype=pl.FP32)
                for i_0, (acc_0,) in pl.range(5, init_values=[init_0]):
                    new_acc_0: pl.Tensor[[64], pl.FP32] = pl.add(acc_0, 1.0)
                    result1_0 = pl.yield_(new_acc_0)
                for j_0, (acc2_0,) in pl.range(3, init_values=[result1_0]):
                    new_acc2_0: pl.Tensor[[64], pl.FP32] = pl.mul(acc2_0, 2.0)
                    result2_0 = pl.yield_(new_acc2_0)
                return result2_0

        After = passes.convert_to_ssa()(Before)
        ir.assert_structural_equal(After, Expected)


# =============================================================================
# Category 3: If Statements (inside loops, since if needs scalar condition)
# =============================================================================


class TestIfStatements:
    """Tests for if statement conversion to SSA with phi nodes."""

    def test_if_in_loop_both_branches(self):
        """if cond: val=mul(...) else: val=add(...) -> phi node for val"""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                init: pl.Tensor[[64], pl.FP32] = pl.create([64], dtype=pl.FP32)
                for i, (acc,) in pl.range(5, init_values=[init]):
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
                init_0: pl.Tensor[[64], pl.FP32] = pl.create([64], dtype=pl.FP32)
                for i_0, (acc_0,) in pl.range(5, init_values=[init_0]):
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
                init: pl.Tensor[[64], pl.FP32] = pl.create([64], dtype=pl.FP32)
                for i, (acc,) in pl.range(3, init_values=[init]):
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
                init_0: pl.Tensor[[64], pl.FP32] = pl.create([64], dtype=pl.FP32)
                for i_0, (acc_0,) in pl.range(3, init_values=[init_0]):
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
                init1: pl.Tensor[[64], pl.FP32] = pl.create([64], dtype=pl.FP32)
                init2: pl.Tensor[[64], pl.FP32] = pl.create([64], dtype=pl.FP32)
                for i, (a, b) in pl.range(5, init_values=[init1, init2]):
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
                init1_0: pl.Tensor[[64], pl.FP32] = pl.create([64], dtype=pl.FP32)
                init2_0: pl.Tensor[[64], pl.FP32] = pl.create([64], dtype=pl.FP32)
                for i_0, (a_0, b_0) in pl.range(5, init_values=[init1_0, init2_0]):
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
# Category 4: Type Preservation
# =============================================================================


class TestTypePreservation:
    """Tests for type preservation during SSA conversion."""

    def test_fp32_type_preserved(self):
        """FP32 tensor type should be preserved after SSA conversion."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                result = pl.add(x, 1.0)
                result = pl.mul(result, 2.0)
                return result

        @pl.program
        class Expected:
            @pl.function(strict_ssa=True)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                result_0: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
                result_1: pl.Tensor[[64], pl.FP32] = pl.mul(result_0, 2.0)
                return result_1

        After = passes.convert_to_ssa()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_fp16_type_preserved(self):
        """FP16 tensor type should be preserved after SSA conversion."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                x: pl.Tensor[[64, 128], pl.FP16],
                y: pl.Tensor[[64, 128], pl.FP16],
            ) -> pl.Tensor[[64, 128], pl.FP16]:
                result: pl.Tensor[[64, 128], pl.FP16] = pl.add(x, y)
                return result

        @pl.program
        class Expected:
            @pl.function(strict_ssa=True)
            def main(
                self,
                x: pl.Tensor[[64, 128], pl.FP16],
                y: pl.Tensor[[64, 128], pl.FP16],
            ) -> pl.Tensor[[64, 128], pl.FP16]:
                result_0: pl.Tensor[[64, 128], pl.FP16] = pl.add(x, y)
                return result_0

        After = passes.convert_to_ssa()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_multidim_shape_preserved(self):
        """Multi-dimensional tensor shape should be preserved."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[32, 64, 128], pl.FP32]) -> pl.Tensor[[32, 64, 128], pl.FP32]:
                result = pl.add(x, 1.0)
                result = pl.mul(result, 2.0)
                return result

        @pl.program
        class Expected:
            @pl.function(strict_ssa=True)
            def main(self, x: pl.Tensor[[32, 64, 128], pl.FP32]) -> pl.Tensor[[32, 64, 128], pl.FP32]:
                result_0: pl.Tensor[[32, 64, 128], pl.FP32] = pl.add(x, 1.0)
                result_1: pl.Tensor[[32, 64, 128], pl.FP32] = pl.mul(result_0, 2.0)
                return result_1

        After = passes.convert_to_ssa()(Before)
        ir.assert_structural_equal(After, Expected)


# =============================================================================
# Category 5: strict_ssa=True Mode (Parser Tests)
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
# Category 6: Pass Pipeline (convert_to_ssa then verify_ssa)
# =============================================================================


class TestPassPipeline:
    """Tests for running convert_to_ssa followed by verify_ssa."""

    def test_convert_then_verify_straight_line(self):
        """convert_to_ssa output should pass verify_ssa for straight-line reassignment."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                result = pl.add(x, 1.0)
                result = pl.mul(result, 2.0)
                return result

        After = passes.convert_to_ssa()(Before)
        result = passes.verify_ssa()(After)
        assert result is not None

    def test_convert_then_verify_with_control_flow(self):
        """convert_to_ssa output should pass verify_ssa for loop + if pattern."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                init: pl.Tensor[[64], pl.FP32] = pl.create([64], dtype=pl.FP32)
                for i, (acc,) in pl.range(5, init_values=[init]):
                    if i == 0:
                        new_val = pl.mul(acc, 2.0)
                        val = pl.yield_(new_val)
                    else:
                        val = pl.yield_(acc)
                    result = pl.yield_(val)
                return result

        After = passes.convert_to_ssa()(Before)
        result = passes.verify_ssa()(After)
        assert result is not None

    def test_already_ssa_passes_verify(self):
        """Already-SSA code converted should still pass verify."""

        @pl.program
        class Before:
            @pl.function(strict_ssa=True)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                a: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
                b: pl.Tensor[[64], pl.FP32] = pl.mul(a, 2.0)
                return b

        After = passes.convert_to_ssa()(Before)
        ir.assert_structural_equal(After, Before)
        result = passes.verify_ssa()(After)
        assert result is not None


# =============================================================================
# Category 7: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and corner scenarios."""

    def test_single_operation_no_reassignment(self):
        """Single operation function - minimal case."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                result: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
                return result

        @pl.program
        class Expected:
            @pl.function(strict_ssa=True)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                result_0: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
                return result_0

        After = passes.convert_to_ssa()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_many_reassignments(self):
        """Many reassignments of the same variable."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                t = pl.add(x, 1.0)
                t = pl.add(t, 2.0)
                t = pl.add(t, 3.0)
                t = pl.add(t, 4.0)
                t = pl.add(t, 5.0)
                return t

        @pl.program
        class Expected:
            @pl.function(strict_ssa=True)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                t_0: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
                t_1: pl.Tensor[[64], pl.FP32] = pl.add(t_0, 2.0)
                t_2: pl.Tensor[[64], pl.FP32] = pl.add(t_1, 3.0)
                t_3: pl.Tensor[[64], pl.FP32] = pl.add(t_2, 4.0)
                t_4: pl.Tensor[[64], pl.FP32] = pl.add(t_3, 5.0)
                return t_4

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

    def test_chain_of_reassignments(self):
        """Chain: result = f(x); result = g(result); ... result = h(result)"""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                result = pl.mul(x, 2.0)
                result = pl.add(result, 1.0)
                result = pl.exp(result)
                result = pl.mul(result, 0.5)
                return result

        @pl.program
        class Expected:
            @pl.function(strict_ssa=True)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                result_0: pl.Tensor[[64], pl.FP32] = pl.mul(x, 2.0)
                result_1: pl.Tensor[[64], pl.FP32] = pl.add(result_0, 1.0)
                result_2: pl.Tensor[[64], pl.FP32] = pl.exp(result_1)
                result_3: pl.Tensor[[64], pl.FP32] = pl.mul(result_2, 0.5)
                return result_3

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
                acc: pl.Tensor[[64], pl.FP32] = pl.create([64], dtype=pl.FP32)
                for i in pl.range(10):
                    acc = pl.add(acc, 1.0)
                return acc

        @pl.program
        class Expected:
            @pl.function(strict_ssa=True)
            def main(self, x_0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                acc_0: pl.Tensor[[64], pl.FP32] = pl.create([64], dtype=pl.FP32)
                for i_0, (acc_iter_1,) in pl.range(0, 10, 1, init_values=[acc_0]):
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
                for i_0, (result_iter_1,) in pl.range(0, 5, 1, init_values=[result_0]):
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
                for i_0, (a_iter_1, b_iter_2) in pl.range(0, 3, 1, init_values=[a_0, b_0]):
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

    def test_backward_compat_explicit_iter_args(self):
        """Backward compatibility: explicit iter_args syntax still works."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                init: pl.Tensor[[64], pl.FP32] = pl.create([64], dtype=pl.FP32)
                for i, (acc,) in pl.range(10, init_values=[init]):
                    new_acc: pl.Tensor[[64], pl.FP32] = pl.add(acc, x)
                    result = pl.yield_(new_acc)
                return result

        @pl.program
        class Expected:
            @pl.function(strict_ssa=True)
            def main(self, x_0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                init_0: pl.Tensor[[64], pl.FP32] = pl.create([64], dtype=pl.FP32)
                for i_0, (acc_0,) in pl.range(0, 10, 1, init_values=[init_0]):
                    new_acc_0: pl.Tensor[[64], pl.FP32] = pl.add(acc_0, x_0)
                    result_0 = pl.yield_(new_acc_0)
                return result_0

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
                for i_0, (result_iter_1,) in pl.range(0, 3, 1, init_values=[result_0]):
                    for j_0, (result_iter_2,) in pl.range(0, 2, 1, init_values=[result_iter_1]):
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

        After = passes.convert_to_ssa()(Before)
        passes.verify_ssa()(After)

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

        After = passes.convert_to_ssa()(Before)
        passes.verify_ssa()(After)

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

        After = passes.convert_to_ssa()(Before)
        passes.verify_ssa()(After)

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

        After = passes.convert_to_ssa()(Before)
        passes.verify_ssa()(After)

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
                for i_0, (result_iter_1,) in pl.range(0, 2, 1, init_values=[result_0]):
                    result_2: pl.Tensor[[64], pl.FP32] = pl.add(result_iter_1, 1.0)
                    result_1 = pl.yield_(result_2)
                for j_0, (result_iter_3,) in pl.range(0, 3, 1, init_values=[result_1]):
                    result_4: pl.Tensor[[64], pl.FP32] = pl.mul(result_iter_3, 1.5)
                    result_3 = pl.yield_(result_4)
                return result_3

        After = passes.convert_to_ssa()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_deeply_nested_loops_plain(self):
        """Three levels of nested loops."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                result: pl.Tensor[[64], pl.FP32] = x
                for i in pl.range(2):
                    for j in pl.range(2):
                        for k in pl.range(2):
                            result = pl.add(result, 1.0)
                return result

        After = passes.convert_to_ssa()(Before)
        passes.verify_ssa()(After)

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

        After = passes.convert_to_ssa()(Before)
        passes.verify_ssa()(After)

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
                for i_0, (result_iter_1,) in pl.range(0, 3, 1, init_values=[result_0]):
                    result_2: pl.Tensor[[64], pl.FP32] = pl.add(result_iter_1, 1.0)
                    result_1 = pl.yield_(result_2)
                final_0: pl.Tensor[[64], pl.FP32] = pl.mul(result_1, 2.0)
                return final_0

        After = passes.convert_to_ssa()(Before)
        ir.assert_structural_equal(After, Expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
