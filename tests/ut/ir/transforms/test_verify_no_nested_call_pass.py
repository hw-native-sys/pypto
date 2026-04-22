# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for NoNestedCall verification rule (via FlattenCallExpr).

NoNestedCallVerifyRule has no direct Python bindings, so these tests exercise
it indirectly: for each nested-call shape, run FlattenCallExpr and assert
the result is structurally equal to a hand-written flat program that has
no nested calls.  If any nested call remains, structural equality fails.
"""

import pypto.language as pl
import pytest
from pypto import ir, passes


def test_nested_call_in_call_args():
    """Nested call in argument position is hoisted to a temporary."""

    @pl.program
    class Before:
        @pl.function
        def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            result: pl.Tensor[[64], pl.FP32] = pl.add(pl.mul(x, 2.0), 1.0)
            return result

    @pl.program
    class Expected:
        @pl.function
        def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            tmp0: pl.Tensor[[64], pl.FP32] = pl.tensor.muls(x, 2.0)
            result: pl.Tensor[[64], pl.FP32] = pl.tensor.adds(tmp0, 1.0)
            return result

    After = passes.flatten_call_expr()(Before)
    ir.assert_structural_equal(After, Expected)


def test_deeply_nested_calls():
    """Three levels of nested calls are each hoisted to a temporary."""

    @pl.program
    class Before:
        @pl.function
        def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            result: pl.Tensor[[64], pl.FP32] = pl.mul(pl.add(pl.exp(x), 1.0), 2.0)
            return result

    @pl.program
    class Expected:
        @pl.function
        def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            tmp0: pl.Tensor[[64], pl.FP32] = pl.tensor.exp(x)
            tmp1: pl.Tensor[[64], pl.FP32] = pl.tensor.adds(tmp0, 1.0)
            result: pl.Tensor[[64], pl.FP32] = pl.tensor.muls(tmp1, 2.0)
            return result

    After = passes.flatten_call_expr()(Before)
    ir.assert_structural_equal(After, Expected)


def test_multiple_nested_calls():
    """Nested calls in multiple argument positions are each hoisted separately."""

    @pl.program
    class Before:
        @pl.function
        def main(
            self,
            x: pl.Tensor[[64], pl.FP32],
            y: pl.Tensor[[64], pl.FP32],
        ) -> pl.Tensor[[64], pl.FP32]:
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
            tmp0: pl.Tensor[[64], pl.FP32] = pl.tensor.muls(x, 2.0)
            tmp1: pl.Tensor[[64], pl.FP32] = pl.tensor.muls(y, 3.0)
            result: pl.Tensor[[64], pl.FP32] = pl.tensor.add(tmp0, tmp1)
            return result

    After = passes.flatten_call_expr()(Before)
    ir.assert_structural_equal(After, Expected)


def test_nested_calls_in_control_flow():
    """Nested calls inside loop/if bodies are flattened in place."""

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
        @pl.function(strict_ssa=True)
        def main(self, x0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            r0: pl.Tensor[[64], pl.FP32] = x0
            for i0, (r1,) in pl.range(5, init_values=(r0,)):
                tmp0: pl.Tensor[[64], pl.FP32] = pl.tensor.muls(r1, 2.0)
                tmp1: pl.Tensor[[64], pl.FP32] = pl.tensor.exp(x0)
                temp0: pl.Tensor[[64], pl.FP32] = pl.tensor.add(tmp0, tmp1)
                if i0 > 2:
                    r2: pl.Tensor[[64], pl.FP32] = temp0
                    r3: pl.Tensor[[64], pl.FP32] = pl.yield_(r2)
                else:
                    r4: pl.Tensor[[64], pl.FP32] = pl.tensor.adds(temp0, 1.0)
                    r3: pl.Tensor[[64], pl.FP32] = pl.yield_(r4)
                r5: pl.Tensor[[64], pl.FP32] = pl.yield_(r3)
            return r5

    After = passes.flatten_call_expr()(passes.convert_to_ssa()(Before))
    ir.assert_structural_equal(After, Expected)


def test_flatten_preserves_flat_code():
    """Already-flat code is returned unchanged."""

    @pl.program
    class Before:
        @pl.function
        def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            temp: pl.Tensor[[64], pl.FP32] = pl.mul(x, 2.0)
            result: pl.Tensor[[64], pl.FP32] = pl.add(temp, 1.0)
            return result

    @pl.program
    class Expected:
        @pl.function
        def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            temp: pl.Tensor[[64], pl.FP32] = pl.tensor.muls(x, 2.0)
            result: pl.Tensor[[64], pl.FP32] = pl.tensor.adds(temp, 1.0)
            return result

    After = passes.flatten_call_expr()(Before)
    ir.assert_structural_equal(After, Expected)


def test_complex_nested_expression_tree():
    """Complex nested tree is flattened into a linear sequence of temporaries."""

    @pl.program
    class Before:
        @pl.function
        def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            a: pl.Tensor[[64], pl.FP32] = pl.mul(pl.exp(x), pl.add(x, 1.0))
            b: pl.Tensor[[64], pl.FP32] = pl.exp(pl.mul(x, 2.0))
            result: pl.Tensor[[64], pl.FP32] = pl.add(a, b)
            return result

    @pl.program
    class Expected:
        @pl.function
        def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            tmp0: pl.Tensor[[64], pl.FP32] = pl.tensor.exp(x)
            tmp1: pl.Tensor[[64], pl.FP32] = pl.tensor.adds(x, 1.0)
            a: pl.Tensor[[64], pl.FP32] = pl.tensor.mul(tmp0, tmp1)
            tmp2: pl.Tensor[[64], pl.FP32] = pl.tensor.muls(x, 2.0)
            b: pl.Tensor[[64], pl.FP32] = pl.tensor.exp(tmp2)
            result: pl.Tensor[[64], pl.FP32] = pl.tensor.add(a, b)
            return result

    After = passes.flatten_call_expr()(Before)
    ir.assert_structural_equal(After, Expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
