# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for parser error handling."""

import pypto
import pypto.language as pl
import pytest
from pypto.language.parser.diagnostics import (
    InvalidOperationError,
    ParserSyntaxError,
    ParserTypeError,
    SSAViolationError,
    UndefinedVariableError,
    UnsupportedFeatureError,
)


class TestErrorCases:
    """Tests for parser error handling and validation."""

    def test_missing_parameter_annotation(self):
        """Test error when parameter lacks type annotation."""

        with pytest.raises(ParserTypeError, match="missing type annotation"):

            @pl.function
            def no_annotation(x):
                return x

    def test_missing_return_annotation(self):
        """Test function without return annotation still works."""

        @pl.function
        def no_return_type(x: pl.Tensor[[64], pl.FP32]):
            result: pl.Tensor[[64], pl.FP32] = pl.op.tensor.mul(x, 2.0)
            return result

        # Should still parse successfully
        assert isinstance(no_return_type, pypto.ir.Function)

    def test_undefined_variable_reference(self):
        """Test error when referencing undefined variable."""

        with pytest.raises(UndefinedVariableError, match="Undefined variable"):

            @pl.function
            def undefined_var(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                result: pl.Tensor[[64], pl.FP32] = pl.op.tensor.add(x, undefined)  # noqa: F821 # type: ignore
                return result

    def test_invalid_tensor_type_syntax(self):
        """Test error on invalid tensor type syntax."""

        with pytest.raises(ParserTypeError):

            @pl.function
            def invalid_type(x: pl.Tensor) -> pl.Tensor[[64], pl.FP32]:  # type: ignore
                return x

    def test_iter_arg_init_values_mismatch(self):
        """Test error when iter_args don't match init_values count."""

        with pytest.raises(ParserSyntaxError, match="Mismatch"):

            @pl.function
            def mismatch(n: pl.Tensor[[1], pl.INT32]) -> pl.Tensor[[1], pl.INT32]:
                init1: pl.Tensor[[1], pl.INT32] = pl.op.tensor.create([1], dtype=pl.INT32)
                init2: pl.Tensor[[1], pl.INT32] = pl.op.tensor.create([1], dtype=pl.INT32)

                # 3 iter_args but only 2 init_values
                for i, (v1, v2, v3) in pl.range(5, init_values=[init1, init2]):
                    out1, out2, out3 = pl.yield_(v1, v2, v3)

                return out1

    def test_unsupported_statement_type(self):
        """Test error on unsupported Python statement."""

        with pytest.raises(UnsupportedFeatureError, match="Unsupported"):

            @pl.function
            def unsupported(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                # Try/except is not supported
                try:
                    result: pl.Tensor[[64], pl.FP32] = pl.op.tensor.add(x, 1.0)
                except:  # noqa: E722
                    result: pl.Tensor[[64], pl.FP32] = x
                return result

    def test_invalid_range_usage(self):
        """Test error when for loop doesn't use pl.range()."""

        with pytest.raises(ParserSyntaxError, match=r"must use pl\.range\(\)"):

            @pl.function
            def invalid_loop(n: pl.Tensor[[1], pl.INT32]) -> pl.Tensor[[1], pl.INT32]:
                result: pl.Tensor[[1], pl.INT32] = n
                # Using Python range() instead of pl.range()
                for i in range(10):
                    result = pl.op.tensor.add(result, 1)
                return result

    def test_invalid_loop_target_format(self):
        """Test error on invalid for loop target format."""

        with pytest.raises(ParserSyntaxError, match="target must be"):

            @pl.function
            def bad_target(n: pl.Tensor[[1], pl.INT32]) -> pl.Tensor[[1], pl.INT32]:
                init: pl.Tensor[[1], pl.INT32] = pl.op.tensor.create([1], dtype=pl.INT32)

                # Missing iter_args tuple
                for i in pl.range(5, init_values=[init]):
                    result = pl.yield_(i)

                return result

    def test_unknown_tensor_operation(self):
        """Test error on unknown tensor operation."""

        with pytest.raises(InvalidOperationError, match="Unknown tensor operation"):

            @pl.function
            def unknown_op(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                # nonexistent_op doesn't exist
                result: pl.Tensor[[64], pl.FP32] = pl.op.tensor.nonexistent_op(x)  # type: ignore
                return result


class TestSSAValidation:
    """Tests for SSA validation."""

    def test_ssa_violation_double_assignment(self):
        """Test that double assignment in same scope is caught with strict_ssa=True."""

        with pytest.raises(SSAViolationError):

            @pl.function(strict_ssa=True)
            def double_assign(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                # First assignment
                result: pl.Tensor[[64], pl.FP32] = pl.op.tensor.mul(x, 2.0)
                # Second assignment (SSA violation)
                result: pl.Tensor[[64], pl.FP32] = pl.op.tensor.add(x, 1.0)
                return result

    def test_variable_from_inner_scope_not_accessible(self):
        """Test that variables from inner scopes aren't accessible without yield."""

        # Note: This test demonstrates the expected behavior
        # The current implementation tracks yields, so this should work correctly
        @pl.function
        def scope_test(n: pl.Tensor[[1], pl.INT32]) -> pl.Tensor[[64], pl.FP32]:
            init: pl.Tensor[[64], pl.FP32] = pl.op.tensor.create([64], dtype=pl.FP32)

            for i, (acc,) in pl.range(5, init_values=[init]):
                temp: pl.Tensor[[64], pl.FP32] = pl.op.tensor.add(acc, 1.0)
                # temp is yielded, so it's accessible as 'result'
                result = pl.yield_(temp)

            # Can return result because it was yielded from loop
            return result

        assert isinstance(scope_test, pypto.ir.Function)


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_function_body(self):
        """Test function with minimal body (just return)."""

        @pl.function
        def minimal(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            return x

        assert isinstance(minimal, pypto.ir.Function)

    def test_single_dimension_tensor(self):
        """Test tensor with single dimension."""

        @pl.function
        def single_dim(x: pl.Tensor[[128], pl.FP32]) -> pl.Tensor[[128], pl.FP32]:
            result: pl.Tensor[[128], pl.FP32] = pl.op.tensor.mul(x, 2.0)
            return result

        assert isinstance(single_dim, pypto.ir.Function)

    def test_three_dimension_tensor(self):
        """Test tensor with three dimensions."""

        @pl.function
        def three_dim(
            x: pl.Tensor[[64, 128, 256], pl.FP32],
        ) -> pl.Tensor[[64, 128, 256], pl.FP32]:
            return x

        assert isinstance(three_dim, pypto.ir.Function)

    def test_loop_with_range_one(self):
        """Test loop that executes only once."""

        @pl.function
        def one_iteration(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            for i, (acc,) in pl.range(1, init_values=[x]):
                result = pl.yield_(acc)

            return result

        assert isinstance(one_iteration, pypto.ir.Function)

    def test_loop_with_start_stop_step(self):
        """Test loop with start, stop, and step parameters."""

        @pl.function
        def custom_range(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            for i, (acc,) in pl.range(2, 10, 2, init_values=[x]):
                new_acc: pl.Tensor[[64], pl.FP32] = pl.op.tensor.add(acc, 1.0)
                result = pl.yield_(new_acc)

            return result

        assert isinstance(custom_range, pypto.ir.Function)

    def test_function_with_many_variables(self):
        """Test function with many local variables."""

        @pl.function
        def many_vars(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            v1: pl.Tensor[[64], pl.FP32] = pl.op.tensor.add(x, 1.0)
            v2: pl.Tensor[[64], pl.FP32] = pl.op.tensor.add(v1, 2.0)
            v3: pl.Tensor[[64], pl.FP32] = pl.op.tensor.add(v2, 3.0)
            v4: pl.Tensor[[64], pl.FP32] = pl.op.tensor.add(v3, 4.0)
            v5: pl.Tensor[[64], pl.FP32] = pl.op.tensor.add(v4, 5.0)
            v6: pl.Tensor[[64], pl.FP32] = pl.op.tensor.add(v5, 6.0)
            v7: pl.Tensor[[64], pl.FP32] = pl.op.tensor.add(v6, 7.0)
            v8: pl.Tensor[[64], pl.FP32] = pl.op.tensor.add(v7, 8.0)
            v9: pl.Tensor[[64], pl.FP32] = pl.op.tensor.add(v8, 9.0)
            v10: pl.Tensor[[64], pl.FP32] = pl.op.tensor.add(v9, 10.0)
            return v10

        assert isinstance(many_vars, pypto.ir.Function)

    def test_different_shape_tensors(self):
        """Test function with tensors of different shapes."""

        @pl.function
        def diff_shapes(
            a: pl.Tensor[[64], pl.FP32],
            b: pl.Tensor[[128], pl.FP32],
            c: pl.Tensor[[256], pl.FP32],
        ) -> pl.Tensor[[64], pl.FP32]:
            return a

        assert isinstance(diff_shapes, pypto.ir.Function)
        assert len(diff_shapes.params) == 3
