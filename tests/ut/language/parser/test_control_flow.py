# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for control flow parsing (for loops, if statements)."""

import pypto.language as pl
from pypto.pypto_core import ir


class TestForLoops:
    """Tests for for loop parsing."""

    def test_simple_for_loop(self):
        """Test simple for loop with one iter_arg."""

        @pl.function
        def sum_loop(n: pl.Tensor[[1], pl.INT32]) -> pl.Tensor[[1], pl.INT32]:
            init: pl.Tensor[[1], pl.INT32] = pl.op.tensor.create([1], dtype=pl.INT32)

            for i, (sum_val,) in pl.range(10, init_values=[init]):
                new_sum: pl.Tensor[[1], pl.INT32] = pl.op.tensor.add(sum_val, i)
                result = pl.yield_(new_sum)

            return result

        assert isinstance(sum_loop, ir.Function)
        assert sum_loop.name == "sum_loop"

    def test_for_loop_multiple_iter_args(self):
        """Test for loop with multiple iteration arguments."""

        @pl.function
        def multi_iter(n: pl.Tensor[[1], pl.INT32]) -> pl.Tensor[[1], pl.INT32]:
            init1: pl.Tensor[[1], pl.INT32] = pl.op.tensor.create([1], dtype=pl.INT32)
            init2: pl.Tensor[[1], pl.INT32] = pl.op.tensor.create([1], dtype=pl.INT32)

            for i, (val1, val2) in pl.range(5, init_values=[init1, init2]):
                new1: pl.Tensor[[1], pl.INT32] = pl.op.tensor.add(val1, i)
                new2: pl.Tensor[[1], pl.INT32] = pl.op.tensor.mul(val2, 2)
                out1, out2 = pl.yield_(new1, new2)

            return out1

        assert isinstance(multi_iter, ir.Function)

    def test_for_loop_with_range_params(self):
        """Test for loop with start, stop, step parameters."""

        @pl.function
        def range_params(n: pl.Tensor[[1], pl.INT32]) -> pl.Tensor[[1], pl.INT32]:
            init: pl.Tensor[[1], pl.INT32] = pl.op.tensor.create([1], dtype=pl.INT32)

            for i, (acc,) in pl.range(0, 10, 2, init_values=[init]):
                new_acc: pl.Tensor[[1], pl.INT32] = pl.op.tensor.add(acc, i)
                result = pl.yield_(new_acc)

            return result

        assert isinstance(range_params, ir.Function)

    def test_nested_for_loops(self):
        """Test nested for loops."""

        @pl.function
        def nested_loops(n: pl.Tensor[[1], pl.INT32]) -> pl.Tensor[[1], pl.INT32]:
            init: pl.Tensor[[1], pl.INT32] = pl.op.tensor.create([1], dtype=pl.INT32)

            for i, (outer,) in pl.range(3, init_values=[init]):
                for j, (inner,) in pl.range(2, init_values=[outer]):
                    new_inner: pl.Tensor[[1], pl.INT32] = pl.op.tensor.add(inner, 1)
                    inner_out = pl.yield_(new_inner)

                outer_out = pl.yield_(inner_out)

            return outer_out

        assert isinstance(nested_loops, ir.Function)

    def test_for_loop_with_operations(self):
        """Test for loop with tensor operations."""

        @pl.function
        def loop_ops(x: pl.Tensor[[64, 128], pl.FP32]) -> pl.Tensor[[64, 128], pl.FP32]:
            init: pl.Tensor[[64, 128], pl.FP32] = pl.op.tensor.create([64, 128], dtype=pl.FP32)

            for i, (acc,) in pl.range(4, init_values=[init]):
                temp: pl.Tensor[[64, 128], pl.FP32] = pl.op.tensor.add(acc, x)
                result = pl.yield_(temp)

            return result

        assert isinstance(loop_ops, ir.Function)


class TestIfStatements:
    """Tests for if statement parsing.

    Note: If conditions currently require scalar types, not tensors.
    Tests use scalar loop variables for conditions.
    """

    def test_if_in_loop(self):
        """Test if statement inside for loop."""

        @pl.function
        def if_in_loop(n: pl.Tensor[[1], pl.INT32]) -> pl.Tensor[[64], pl.FP32]:
            init: pl.Tensor[[64], pl.FP32] = pl.op.tensor.create([64], dtype=pl.FP32)

            for i, (acc,) in pl.range(5, init_values=[init]):
                if i == 0:
                    new_val: pl.Tensor[[64], pl.FP32] = pl.op.tensor.mul(acc, 2.0)
                    val: pl.Tensor[[64], pl.FP32] = pl.yield_(new_val)
                else:
                    val: pl.Tensor[[64], pl.FP32] = pl.yield_(acc)

                result = pl.yield_(val)

            return result

        assert isinstance(if_in_loop, ir.Function)


class TestComplexControlFlow:
    """Tests for complex control flow combinations."""

    def test_loop_with_if_and_multiple_iter_args(self):
        """Test loop with if statement and multiple iter_args."""

        @pl.function
        def complex_flow(n: pl.Tensor[[1], pl.INT32]) -> pl.Tensor[[64], pl.FP32]:
            acc1: pl.Tensor[[64], pl.FP32] = pl.op.tensor.create([64], dtype=pl.FP32)
            acc2: pl.Tensor[[64], pl.FP32] = pl.op.tensor.create([64], dtype=pl.FP32)

            for i, (a1, a2) in pl.range(10, init_values=[acc1, acc2]):
                if i == 0:
                    new1: pl.Tensor[[64], pl.FP32] = pl.op.tensor.mul(a1, 2.0)
                    new2: pl.Tensor[[64], pl.FP32] = pl.op.tensor.mul(a2, 3.0)
                    val1, val2 = pl.yield_(new1, new2)
                else:
                    val1, val2 = pl.yield_(a1, a2)

                out1, out2 = pl.yield_(val1, val2)

            return out1

        assert isinstance(complex_flow, ir.Function)

    def test_sequential_loops(self):
        """Test sequential (not nested) for loops."""

        @pl.function
        def sequential_loops(n: pl.Tensor[[1], pl.INT32]) -> pl.Tensor[[64], pl.FP32]:
            init: pl.Tensor[[64], pl.FP32] = pl.op.tensor.create([64], dtype=pl.FP32)

            # First loop
            for i, (acc,) in pl.range(5, init_values=[init]):
                new_acc: pl.Tensor[[64], pl.FP32] = pl.op.tensor.add(acc, 1.0)
                result1 = pl.yield_(new_acc)

            # Second loop uses output of first
            for j, (acc2,) in pl.range(3, init_values=[result1]):
                new_acc2: pl.Tensor[[64], pl.FP32] = pl.op.tensor.mul(acc2, 2.0)
                result2 = pl.yield_(new_acc2)

            return result2

        assert isinstance(sequential_loops, ir.Function)

    def test_loop_without_iter_args(self):
        """Test loop without iter_args."""

        @pl.function
        def loop_without_iter_args(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            result: pl.Tensor[[64], pl.FP32] = x
            for i in pl.range(3):
                if i > 0:
                    temp = pl.op.tensor.mul(result, 2.0)
                    result = temp
                else:
                    temp = pl.op.tensor.add(result, 1.0)
                    result = temp
            return result

        assert isinstance(loop_without_iter_args, ir.Function)
