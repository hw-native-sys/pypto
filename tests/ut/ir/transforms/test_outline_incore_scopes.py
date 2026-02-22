# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for OutlineIncoreScopes pass."""

import pypto.language as pl
from pypto import ir, passes


class TestOutlineIncoreScopes:
    """Test OutlineIncoreScopes pass."""

    def test_outline_simple_incore_scope(self):
        """Test outlining a simple InCore scope."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.incore():
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = self.main_incore_0(x)
                return y

        # Convert to SSA first (required by outline pass)
        Before = passes.convert_to_ssa()(Before)
        Expected = passes.convert_to_ssa()(Expected)

        # Apply outline pass
        After = passes.outline_incore_scopes()(Before)

        # Should be structurally equal
        ir.assert_structural_equal(After, Expected)

    def test_outline_multiple_incore_scopes(self):
        """Test outlining multiple InCore scopes in one function."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.incore():
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                with pl.incore():
                    z: pl.Tensor[[64], pl.FP32] = pl.mul(y, y)
                return z

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_1(self, y: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                z: pl.Tensor[[64], pl.FP32] = pl.mul(y, y)
                return z

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = self.main_incore_0(x)
                z: pl.Tensor[[64], pl.FP32] = self.main_incore_1(y)
                return z

        # Convert to SSA first
        Before = passes.convert_to_ssa()(Before)
        Expected = passes.convert_to_ssa()(Expected)

        # Apply outline pass
        After = passes.outline_incore_scopes()(Before)

        # Should be structurally equal
        ir.assert_structural_equal(After, Expected)

    def test_outline_preserves_non_incore_functions(self):
        """Test that non-InCore functions are preserved unchanged."""

        @pl.program
        class Before:
            @pl.function
            def helper(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                result: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return result

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.incore():
                    y: pl.Tensor[[64], pl.FP32] = pl.mul(x, x)
                return y

        @pl.program
        class Expected:
            @pl.function
            def helper(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                result: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return result

            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = pl.mul(x, x)
                return y

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = self.main_incore_0(x)
                return y

        # Convert to SSA first
        Before = passes.convert_to_ssa()(Before)
        Expected = passes.convert_to_ssa()(Expected)

        # Apply outline pass
        After = passes.outline_incore_scopes()(Before)

        # Should be structurally equal
        ir.assert_structural_equal(After, Expected)

    def test_outline_scope_with_multiple_inputs(self):
        """Test outlining scope that uses multiple outer variables."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self, x: pl.Tensor[[64], pl.FP32], y: pl.Tensor[[64], pl.FP32]
            ) -> pl.Tensor[[64], pl.FP32]:
                a: pl.Tensor[[64], pl.FP32] = pl.add(x, y)
                b: pl.Tensor[[64], pl.FP32] = pl.mul(x, y)
                with pl.incore():
                    result: pl.Tensor[[64], pl.FP32] = pl.add(a, b)
                return result

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self, a: pl.Tensor[[64], pl.FP32], b: pl.Tensor[[64], pl.FP32]
            ) -> pl.Tensor[[64], pl.FP32]:
                result: pl.Tensor[[64], pl.FP32] = pl.add(a, b)
                return result

            @pl.function
            def main(
                self, x: pl.Tensor[[64], pl.FP32], y: pl.Tensor[[64], pl.FP32]
            ) -> pl.Tensor[[64], pl.FP32]:
                a: pl.Tensor[[64], pl.FP32] = pl.add(x, y)
                b: pl.Tensor[[64], pl.FP32] = pl.mul(x, y)
                result: pl.Tensor[[64], pl.FP32] = self.main_incore_0(a, b)
                return result

        Before = passes.convert_to_ssa()(Before)
        Expected = passes.convert_to_ssa()(Expected)
        After = passes.outline_incore_scopes()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_outline_scope_with_multiple_outputs(self):
        """Test outlining scope that produces multiple values.

        The Before/After pattern can't express TupleGetItem in the DSL,
        so we verify properties directly.
        """

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.incore():
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                    z: pl.Tensor[[64], pl.FP32] = pl.mul(x, x)
                result: pl.Tensor[[64], pl.FP32] = pl.add(y, z)
                return result

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self, x: pl.Tensor[[64], pl.FP32]
            ) -> tuple[pl.Tensor[[64], pl.FP32], pl.Tensor[[64], pl.FP32]]:
                y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                z: pl.Tensor[[64], pl.FP32] = pl.mul(x, x)
                return (y, z)

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                ret = self.main_incore_0(x)
                y = ret[0]
                z = ret[1]
                result: pl.Tensor[[64], pl.FP32] = pl.add(y, z)
                return result

        Before = passes.convert_to_ssa()(Before)
        After = passes.outline_incore_scopes()(Before)

        ir.assert_structural_equal(After, Expected)

    def test_outline_nested_incore_scopes(self):
        """Test outlining nested InCore scopes."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.incore():
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                    with pl.incore():
                        z: pl.Tensor[[64], pl.FP32] = pl.mul(y, y)
                return z

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0_incore_0(self, y: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                z: pl.Tensor[[64], pl.FP32] = pl.mul(y, y)
                return z

            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                z: pl.Tensor[[64], pl.FP32] = self.main_incore_0_incore_0(y)
                return z

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                z: pl.Tensor[[64], pl.FP32] = self.main_incore_0(x)
                return z

        Before = passes.convert_to_ssa()(Before)
        Expected = passes.convert_to_ssa()(Expected)
        After = passes.outline_incore_scopes()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_outline_scope_with_single_input_single_output(self):
        """Test outlining scope with simple single input/output."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                a: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                with pl.incore():
                    y: pl.Tensor[[64], pl.FP32] = pl.mul(a, a)
                result: pl.Tensor[[64], pl.FP32] = pl.add(y, y)
                return result

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(self, a: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = pl.mul(a, a)
                return y

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                a: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                y: pl.Tensor[[64], pl.FP32] = self.main_incore_0(a)
                result: pl.Tensor[[64], pl.FP32] = pl.add(y, y)
                return result

        Before = passes.convert_to_ssa()(Before)
        Expected = passes.convert_to_ssa()(Expected)
        After = passes.outline_incore_scopes()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_outline_multiple_functions_with_scopes(self):
        """Test outlining scopes in multiple functions (independent numbering)."""

        @pl.program
        class Before:
            @pl.function
            def func1(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.incore():
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

            @pl.function
            def func2(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.incore():
                    y: pl.Tensor[[64], pl.FP32] = pl.mul(x, x)
                return y

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def func1_incore_0(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

            @pl.function
            def func1(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = self.func1_incore_0(x)
                return y

            @pl.function(type=pl.FunctionType.InCore)
            def func2_incore_0(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = pl.mul(x, x)
                return y

            @pl.function
            def func2(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = self.func2_incore_0(x)
                return y

        Before = passes.convert_to_ssa()(Before)
        Expected = passes.convert_to_ssa()(Expected)
        After = passes.outline_incore_scopes()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_outline_scope_in_control_flow(self):
        """Test outlining scope inside conditional statement."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32], cond: pl.Scalar[pl.BOOL]) -> pl.Tensor[[64], pl.FP32]:
                if cond:
                    with pl.incore():
                        y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)  # type: ignore[no-redef]
                else:
                    y: pl.Tensor[[64], pl.FP32] = pl.mul(x, x)  # type: ignore[no-redef,unreachable]
                return y

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32], cond: pl.Scalar[pl.BOOL]) -> pl.Tensor[[64], pl.FP32]:
                if cond:
                    y: pl.Tensor[[64], pl.FP32] = self.main_incore_0(x)  # type: ignore[no-redef]
                else:
                    y: pl.Tensor[[64], pl.FP32] = pl.mul(x, x)  # type: ignore[no-redef,unreachable]
                return y

        Before = passes.convert_to_ssa()(Before)
        Expected = passes.convert_to_ssa()(Expected)
        After = passes.outline_incore_scopes()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_outline_incore_with_if_yield(self):
        """Test outline_incore_scopes with IfStmt containing unannotated yields (issue #233)."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32], cond: pl.Scalar[pl.BOOL]) -> pl.Tensor[[64], pl.FP32]:
                with pl.incore():
                    if cond:
                        y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                        z = pl.yield_(y)  # Unannotated - should infer type
                    else:
                        y2: pl.Tensor[[64], pl.FP32] = pl.mul(x, x)
                        z = pl.yield_(y2)
                return z

        Before = passes.convert_to_ssa()(Before)
        After = passes.outline_incore_scopes()(Before)

        printed = ir.python_print(After)
        # The outlined incore function should have correct return type, not Tensor[[1], INT32]
        assert "Tensor[[1], pl.INT32]" not in printed
        assert "Tensor[[64], pl.FP32]" in printed

    def test_outline_scope_with_intermediate_computation(self):
        """Test outlining scope with computation before, inside, and after."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                a: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                b: pl.Tensor[[64], pl.FP32] = pl.mul(a, a)
                with pl.incore():
                    c: pl.Tensor[[64], pl.FP32] = pl.add(b, b)
                    d: pl.Tensor[[64], pl.FP32] = pl.mul(c, c)
                e: pl.Tensor[[64], pl.FP32] = pl.add(d, d)
                return e

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(self, b: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                c: pl.Tensor[[64], pl.FP32] = pl.add(b, b)
                d: pl.Tensor[[64], pl.FP32] = pl.mul(c, c)
                return d

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                a: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                b: pl.Tensor[[64], pl.FP32] = pl.mul(a, a)
                d: pl.Tensor[[64], pl.FP32] = self.main_incore_0(b)
                e: pl.Tensor[[64], pl.FP32] = pl.add(d, d)
                return e

        Before = passes.convert_to_ssa()(Before)
        Expected = passes.convert_to_ssa()(Expected)
        After = passes.outline_incore_scopes()(Before)
        ir.assert_structural_equal(After, Expected)
