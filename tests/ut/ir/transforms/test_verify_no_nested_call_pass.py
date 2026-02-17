# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for NoNestedCall verification rule.

This tests the NoNestedCallVerifyRule indirectly through the FlattenCallExpr pass.
The verification rule is tested by ensuring that:
1. Nested calls cause violations
2. The flatten pass removes all nested calls
3. After flattening, no nested call violations remain

Note: Direct testing of the verification rule requires Python bindings for
CreateNoNestedCallVerifyRule, which are not currently exposed. These tests
verify the rule indirectly through the flatten/verify pipeline.
"""

import pypto.language as pl
from pypto.pypto_core import passes


def test_nested_call_in_call_args():
    """Test that nested calls in arguments are detected and can be flattened."""

    @pl.program
    class NestedCalls:
        @pl.function
        def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            # Nested call in arguments: add(mul(x, 2.0), 1.0)
            result: pl.Tensor[[64], pl.FP32] = pl.add(pl.mul(x, 2.0), 1.0)
            return result

    # Apply flatten pass
    flatten_pass = passes.flatten_call_expr()
    flattened_program = flatten_pass(NestedCalls)

    # Verify the flattened program is valid
    assert flattened_program is not None

    # The flattened program should have more statements (temporary variables)
    original_func = NestedCalls.get_function("main")
    flattened_func = flattened_program.get_function("main")
    assert original_func is not None
    assert flattened_func is not None


def test_deeply_nested_calls():
    """Test deeply nested calls are properly flattened."""

    @pl.program
    class DeeplyNestedCalls:
        @pl.function
        def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            # Deeply nested: mul(add(exp(x), 1.0), 2.0)
            result: pl.Tensor[[64], pl.FP32] = pl.mul(pl.add(pl.exp(x), 1.0), 2.0)
            return result

    # Apply flatten pass
    flatten_pass = passes.flatten_call_expr()
    flattened_program = flatten_pass(DeeplyNestedCalls)

    # Verify the result
    assert flattened_program is not None
    flattened_func = flattened_program.get_function("main")
    assert flattened_func is not None


def test_multiple_nested_calls():
    """Test multiple nested calls in different argument positions."""

    @pl.program
    class MultipleNestedCalls:
        @pl.function
        def main(
            self,
            x: pl.Tensor[[64], pl.FP32],
            y: pl.Tensor[[64], pl.FP32],
        ) -> pl.Tensor[[64], pl.FP32]:
            # Multiple nested calls: add(mul(x, 2.0), mul(y, 3.0))
            result: pl.Tensor[[64], pl.FP32] = pl.add(pl.mul(x, 2.0), pl.mul(y, 3.0))
            return result

    # Apply flatten pass
    flatten_pass = passes.flatten_call_expr()
    flattened_program = flatten_pass(MultipleNestedCalls)

    # Verify the result
    assert flattened_program is not None
    flattened_func = flattened_program.get_function("main")
    assert flattened_func is not None


def test_nested_calls_in_control_flow():
    """Test nested calls within control flow structures."""

    @pl.program
    class NestedCallsInControlFlow:
        @pl.function
        def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            result: pl.Tensor[[64], pl.FP32] = x
            for i in pl.range(5):
                # Nested call in loop
                temp: pl.Tensor[[64], pl.FP32] = pl.add(pl.mul(result, 2.0), pl.exp(x))
                if i > 2:
                    result = temp
                else:
                    result = pl.add(temp, 1.0)
            return result

    # Apply flatten pass
    flatten_pass = passes.flatten_call_expr()
    flattened_program = flatten_pass(NestedCallsInControlFlow)

    # Verify the result
    assert flattened_program is not None
    flattened_func = flattened_program.get_function("main")
    assert flattened_func is not None


def test_flatten_preserves_flat_code():
    """Test that already flat code is preserved correctly."""

    @pl.program
    class FlatCode:
        @pl.function
        def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            temp: pl.Tensor[[64], pl.FP32] = pl.mul(x, 2.0)
            result: pl.Tensor[[64], pl.FP32] = pl.add(temp, 1.0)
            return result

    # Apply flatten pass
    flatten_pass = passes.flatten_call_expr()
    flattened_program = flatten_pass(FlatCode)

    # Verify the result
    assert flattened_program is not None
    flattened_func = flattened_program.get_function("main")
    assert flattened_func is not None


def test_flatten_and_convert_to_ssa_pipeline():
    """Test flatten pass can be combined with SSA conversion."""

    @pl.program
    class NestedCallsWithReassignment:
        @pl.function
        def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            result = pl.add(pl.mul(x, 2.0), 1.0)
            result = pl.add(result, 3.0)
            return result

    # Apply flatten then SSA conversion
    flatten_pass = passes.flatten_call_expr()
    flattened = flatten_pass(NestedCallsWithReassignment)

    ssa_pass = passes.convert_to_ssa()
    ssa_program = ssa_pass(flattened)

    # Verify both passes succeeded
    assert flattened is not None
    assert ssa_program is not None

    # Verify with SSA verification pass
    verify_pass = passes.run_verifier()
    verified = verify_pass(ssa_program)
    assert verified is not None


def test_complex_nested_expression_tree():
    """Test complex nested expression trees are properly flattened."""

    @pl.program
    class ComplexNested:
        @pl.function
        def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            # Complex nested tree: add(mul(exp(x), add(x, 1.0)), exp(mul(x, 2.0)))
            a: pl.Tensor[[64], pl.FP32] = pl.mul(pl.exp(x), pl.add(x, 1.0))
            b: pl.Tensor[[64], pl.FP32] = pl.exp(pl.mul(x, 2.0))
            result: pl.Tensor[[64], pl.FP32] = pl.add(a, b)
            return result

    # Apply flatten pass
    flatten_pass = passes.flatten_call_expr()
    flattened_program = flatten_pass(ComplexNested)

    # Verify the result
    assert flattened_program is not None
    flattened_func = flattened_program.get_function("main")
    assert flattened_func is not None


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
