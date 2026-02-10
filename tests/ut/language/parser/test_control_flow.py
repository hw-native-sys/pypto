# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for control flow parsing (for loops, if statements)."""

import pypto
import pypto.language as pl
import pytest
from pypto.pypto_core import ir


class TestForLoops:
    """Tests for for loop parsing."""

    def test_simple_for_loop(self):
        """Test simple for loop with one iter_arg."""

        @pl.function
        def sum_loop(n: pl.Tensor[[1], pl.INT32]) -> pl.Tensor[[1], pl.INT32]:
            init: pl.Tensor[[1], pl.INT32] = pl.create([1], dtype=pl.INT32)

            for i, (sum_val,) in pl.range(10, init_values=[init]):
                new_sum: pl.Tensor[[1], pl.INT32] = pl.add(sum_val, i)
                result = pl.yield_(new_sum)

            return result

        assert isinstance(sum_loop, ir.Function)
        assert sum_loop.name == "sum_loop"

    def test_for_loop_multiple_iter_args(self):
        """Test for loop with multiple iteration arguments."""

        @pl.function
        def multi_iter(n: pl.Tensor[[1], pl.INT32]) -> pl.Tensor[[1], pl.INT32]:
            init1: pl.Tensor[[1], pl.INT32] = pl.create([1], dtype=pl.INT32)
            init2: pl.Tensor[[1], pl.INT32] = pl.create([1], dtype=pl.INT32)

            for i, (val1, val2) in pl.range(5, init_values=[init1, init2]):
                new1: pl.Tensor[[1], pl.INT32] = pl.add(val1, i)
                new2: pl.Tensor[[1], pl.INT32] = pl.mul(val2, 2)
                out1, out2 = pl.yield_(new1, new2)

            return out1

        assert isinstance(multi_iter, ir.Function)

    def test_for_loop_with_range_params(self):
        """Test for loop with start, stop, step parameters."""

        @pl.function
        def range_params(n: pl.Tensor[[1], pl.INT32]) -> pl.Tensor[[1], pl.INT32]:
            init: pl.Tensor[[1], pl.INT32] = pl.create([1], dtype=pl.INT32)

            for i, (acc,) in pl.range(0, 10, 2, init_values=[init]):
                new_acc: pl.Tensor[[1], pl.INT32] = pl.add(acc, i)
                result = pl.yield_(new_acc)

            return result

        assert isinstance(range_params, ir.Function)

    def test_nested_for_loops(self):
        """Test nested for loops."""

        @pl.function
        def nested_loops(n: pl.Tensor[[1], pl.INT32]) -> pl.Tensor[[1], pl.INT32]:
            init: pl.Tensor[[1], pl.INT32] = pl.create([1], dtype=pl.INT32)

            for i, (outer,) in pl.range(3, init_values=[init]):
                for j, (inner,) in pl.range(2, init_values=[outer]):
                    new_inner: pl.Tensor[[1], pl.INT32] = pl.add(inner, 1)
                    inner_out = pl.yield_(new_inner)

                outer_out = pl.yield_(inner_out)

            return outer_out

        assert isinstance(nested_loops, ir.Function)

    def test_for_loop_with_operations(self):
        """Test for loop with tensor operations."""

        @pl.function
        def loop_ops(x: pl.Tensor[[64, 128], pl.FP32]) -> pl.Tensor[[64, 128], pl.FP32]:
            init: pl.Tensor[[64, 128], pl.FP32] = pl.create([64, 128], dtype=pl.FP32)

            for i, (acc,) in pl.range(4, init_values=[init]):
                temp: pl.Tensor[[64, 128], pl.FP32] = pl.add(acc, x)
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
            init: pl.Tensor[[64], pl.FP32] = pl.create([64], dtype=pl.FP32)

            for i, (acc,) in pl.range(5, init_values=[init]):
                if i == 0:
                    new_val: pl.Tensor[[64], pl.FP32] = pl.mul(acc, 2.0)
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
            acc1: pl.Tensor[[64], pl.FP32] = pl.create([64], dtype=pl.FP32)
            acc2: pl.Tensor[[64], pl.FP32] = pl.create([64], dtype=pl.FP32)

            for i, (a1, a2) in pl.range(10, init_values=[acc1, acc2]):
                if i == 0:
                    new1: pl.Tensor[[64], pl.FP32] = pl.mul(a1, 2.0)
                    new2: pl.Tensor[[64], pl.FP32] = pl.mul(a2, 3.0)
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
            init: pl.Tensor[[64], pl.FP32] = pl.create([64], dtype=pl.FP32)

            # First loop
            for i, (acc,) in pl.range(5, init_values=[init]):
                new_acc: pl.Tensor[[64], pl.FP32] = pl.add(acc, 1.0)
                result1 = pl.yield_(new_acc)

            # Second loop uses output of first
            for j, (acc2,) in pl.range(3, init_values=[result1]):
                new_acc2: pl.Tensor[[64], pl.FP32] = pl.mul(acc2, 2.0)
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
                    temp = pl.mul(result, 2.0)
                    result = temp
                else:
                    temp = pl.add(result, 1.0)
                    result = temp
            return result

        assert isinstance(loop_without_iter_args, ir.Function)


class TestParallelForLoops:
    """Tests for parallel for loop parsing with pl.parallel()."""

    def test_simple_parallel_for_loop(self):
        """Test simple parallel for loop with one iter_arg."""

        @pl.function
        def parallel_sum(n: pl.Tensor[[1], pl.INT32]) -> pl.Tensor[[1], pl.INT32]:
            init: pl.Tensor[[1], pl.INT32] = pl.create([1], dtype=pl.INT32)

            for i, (sum_val,) in pl.parallel(10, init_values=[init]):
                new_sum: pl.Tensor[[1], pl.INT32] = pl.add(sum_val, i)
                result = pl.yield_(new_sum)

            return result

        assert isinstance(parallel_sum, ir.Function)
        assert parallel_sum.name == "parallel_sum"

    def test_parallel_for_loop_without_iter_args(self):
        """Test parallel for loop without iter_args."""

        @pl.function
        def parallel_no_iter(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            result: pl.Tensor[[64], pl.FP32] = x
            for i in pl.parallel(3):
                temp = pl.mul(result, 2.0)
                result = temp
            return result

        assert isinstance(parallel_no_iter, ir.Function)

    def test_parallel_for_loop_with_range_params(self):
        """Test parallel for loop with start, stop, step parameters."""

        @pl.function
        def parallel_range(n: pl.Tensor[[1], pl.INT32]) -> pl.Tensor[[1], pl.INT32]:
            init: pl.Tensor[[1], pl.INT32] = pl.create([1], dtype=pl.INT32)

            for i, (acc,) in pl.parallel(0, 10, 2, init_values=[init]):
                new_acc: pl.Tensor[[1], pl.INT32] = pl.add(acc, i)
                result = pl.yield_(new_acc)

            return result

        assert isinstance(parallel_range, ir.Function)

    def test_parallel_for_produces_parallel_kind(self):
        """Test that pl.parallel() produces ForKind.Parallel in the IR."""

        @pl.function
        def par_func(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            init: pl.Tensor[[64], pl.FP32] = pl.create([64], dtype=pl.FP32)
            for i, (acc,) in pl.parallel(10, init_values=[init]):
                new_acc: pl.Tensor[[64], pl.FP32] = pl.add(acc, x)
                result = pl.yield_(new_acc)
            return result

        # Find the ForStmt in the function body
        body = par_func.body
        if isinstance(body, ir.SeqStmts):
            for_stmt = None
            for stmt in body.stmts:
                if isinstance(stmt, ir.ForStmt):
                    for_stmt = stmt
                    break
        elif isinstance(body, ir.ForStmt):
            for_stmt = body
        else:
            for_stmt = None

        assert for_stmt is not None, "Expected ForStmt in function body"
        assert for_stmt.kind == ir.ForKind.Parallel

    def test_range_for_produces_sequential_kind(self):
        """Test that pl.range() produces ForKind.Sequential in the IR."""

        @pl.function
        def seq_func(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            init: pl.Tensor[[64], pl.FP32] = pl.create([64], dtype=pl.FP32)
            for i, (acc,) in pl.range(10, init_values=[init]):
                new_acc: pl.Tensor[[64], pl.FP32] = pl.add(acc, x)
                result = pl.yield_(new_acc)
            return result

        # Find the ForStmt in the function body
        body = seq_func.body
        if isinstance(body, ir.SeqStmts):
            for_stmt = None
            for stmt in body.stmts:
                if isinstance(stmt, ir.ForStmt):
                    for_stmt = stmt
                    break
        elif isinstance(body, ir.ForStmt):
            for_stmt = body
        else:
            for_stmt = None

        assert for_stmt is not None, "Expected ForStmt in function body"
        assert for_stmt.kind == ir.ForKind.Sequential

    def test_parallel_for_printer_output(self):
        """Test that parallel for loop prints with pl.parallel()."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                init: pl.Tensor[[64], pl.FP32] = pl.create([64], dtype=pl.FP32)
                for i, (acc,) in pl.parallel(10, init_values=[init]):
                    new_acc: pl.Tensor[[64], pl.FP32] = pl.add(acc, x)
                    result = pl.yield_(new_acc)
                return result

        printed = pypto.ir.python_print(Before)
        assert "pl.parallel(" in printed
        assert "pl.range(" not in printed

    def test_sequential_for_printer_no_parallel(self):
        """Test that sequential for loop does not print pl.parallel()."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                init: pl.Tensor[[64], pl.FP32] = pl.create([64], dtype=pl.FP32)
                for i, (acc,) in pl.range(10, init_values=[init]):
                    new_acc: pl.Tensor[[64], pl.FP32] = pl.add(acc, x)
                    result = pl.yield_(new_acc)
                return result

        printed = pypto.ir.python_print(Before)
        assert "pl.parallel(" not in printed
        assert "pl.range(" in printed

    def test_parallel_for_structural_not_equal_to_sequential(self):
        """Test that parallel and sequential loops with same body are not structurally equal."""

        @pl.program
        class ParallelProg:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                init: pl.Tensor[[64], pl.FP32] = pl.create([64], dtype=pl.FP32)
                for i, (acc,) in pl.parallel(10, init_values=[init]):
                    new_acc: pl.Tensor[[64], pl.FP32] = pl.add(acc, x)
                    result = pl.yield_(new_acc)
                return result

        @pl.program
        class SequentialProg:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                init: pl.Tensor[[64], pl.FP32] = pl.create([64], dtype=pl.FP32)
                for i, (acc,) in pl.range(10, init_values=[init]):
                    new_acc: pl.Tensor[[64], pl.FP32] = pl.add(acc, x)
                    result = pl.yield_(new_acc)
                return result

        assert not ir.structural_equal(ParallelProg, SequentialProg)

    def test_invalid_iterator_rejected(self):
        """Test that invalid iterator (not range or parallel) is rejected."""
        with pytest.raises(Exception):

            @pl.function
            def bad_func(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                for i in pl.yield_(10):  # type: ignore
                    pass
                return x


def _find_for_stmt(func: ir.Function) -> ir.ForStmt:
    """Helper to extract the first ForStmt from a function body."""
    body = func.body
    if isinstance(body, ir.ForStmt):
        return body
    if isinstance(body, ir.SeqStmts):
        for stmt in body.stmts:
            if isinstance(stmt, ir.ForStmt):
                return stmt
    raise AssertionError("No ForStmt found in function body")


class TestScalarRange:
    """Tests for pl.range() with Scalar type arguments."""

    def test_scalar_param_as_stop(self):
        """Test pl.range(n) where n is a Scalar[INT64] parameter."""

        @pl.function
        def scalar_stop(n: pl.Scalar[pl.INT64], x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            for i in pl.range(n):
                y: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
            return y

        assert isinstance(scalar_stop, ir.Function)
        for_stmt = _find_for_stmt(scalar_stop)
        # stop should be a Var reference to the Scalar parameter 'n'
        assert isinstance(for_stmt.stop, ir.Var)
        assert for_stmt.stop.name == "n"
        assert isinstance(for_stmt.stop.type, ir.ScalarType)

    def test_scalar_param_as_start_stop(self):
        """Test pl.range(0, n) where n is a Scalar[INT64] parameter."""

        @pl.function
        def scalar_start_stop(
            n: pl.Scalar[pl.INT64], x: pl.Tensor[[64], pl.FP32]
        ) -> pl.Tensor[[64], pl.FP32]:
            for i in pl.range(0, n):
                y: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
            return y

        assert isinstance(scalar_start_stop, ir.Function)
        for_stmt = _find_for_stmt(scalar_start_stop)
        assert isinstance(for_stmt.start, ir.ConstInt)
        assert isinstance(for_stmt.stop, ir.Var)
        assert for_stmt.stop.name == "n"

    def test_scalar_param_as_start_stop_step(self):
        """Test pl.range(0, n, s) where n and s are Scalar[INT64] parameters."""

        @pl.function
        def scalar_full_range(
            n: pl.Scalar[pl.INT64],
            s: pl.Scalar[pl.INT64],
            x: pl.Tensor[[64], pl.FP32],
        ) -> pl.Tensor[[64], pl.FP32]:
            for i in pl.range(0, n, s):
                y: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
            return y

        assert isinstance(scalar_full_range, ir.Function)
        for_stmt = _find_for_stmt(scalar_full_range)
        assert isinstance(for_stmt.start, ir.ConstInt)
        assert isinstance(for_stmt.stop, ir.Var)
        assert for_stmt.stop.name == "n"
        assert isinstance(for_stmt.step, ir.Var)
        assert for_stmt.step.name == "s"

    def test_scalar_expression_as_stop(self):
        """Test pl.range(n * 2) where n is a Scalar[INT64] parameter."""

        @pl.function
        def scalar_expr_stop(n: pl.Scalar[pl.INT64], x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            for i in pl.range(n * 2):  # type: ignore[operator]
                y: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
            return y

        assert isinstance(scalar_expr_stop, ir.Function)
        for_stmt = _find_for_stmt(scalar_expr_stop)
        # stop should be a Mul expression (n * 2)
        assert isinstance(for_stmt.stop, ir.Mul)

    def test_scalar_complex_expression_as_stop(self):
        """Test pl.range(n * 2 + 1) where n is a Scalar[INT64] parameter."""

        @pl.function
        def scalar_complex_expr(
            n: pl.Scalar[pl.INT64], x: pl.Tensor[[64], pl.FP32]
        ) -> pl.Tensor[[64], pl.FP32]:
            for i in pl.range(n * 2 + 1):  # type: ignore[operator]
                y: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
            return y

        assert isinstance(scalar_complex_expr, ir.Function)
        for_stmt = _find_for_stmt(scalar_complex_expr)
        # stop should be an Add expression ((n * 2) + 1)
        assert isinstance(for_stmt.stop, ir.Add)

    def test_scalar_floordiv_expression_as_stop(self):
        """Test pl.range(n // 4) where n is a Scalar[INT64] parameter."""

        @pl.function
        def scalar_floordiv_expr(
            n: pl.Scalar[pl.INT64], x: pl.Tensor[[64], pl.FP32]
        ) -> pl.Tensor[[64], pl.FP32]:
            for i in pl.range(n // 4):  # type: ignore[operator]
                y: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
            return y

        assert isinstance(scalar_floordiv_expr, ir.Function)
        for_stmt = _find_for_stmt(scalar_floordiv_expr)
        # stop should be a FloorDiv expression (n // 4)
        assert isinstance(for_stmt.stop, ir.FloorDiv)

    def test_scalar_range_with_iter_args(self):
        """Test pl.range(n, init_values=[init]) where n is a Scalar[INT64] parameter."""

        @pl.function
        def scalar_range_iter(
            n: pl.Scalar[pl.INT64], x: pl.Tensor[[64], pl.FP32]
        ) -> pl.Tensor[[64], pl.FP32]:
            init: pl.Tensor[[64], pl.FP32] = pl.create([64], dtype=pl.FP32)
            for i, (acc,) in pl.range(n, init_values=[init]):
                new_acc: pl.Tensor[[64], pl.FP32] = pl.add(acc, x)
                result = pl.yield_(new_acc)
            return result

        assert isinstance(scalar_range_iter, ir.Function)
        for_stmt = _find_for_stmt(scalar_range_iter)
        assert isinstance(for_stmt.stop, ir.Var)
        assert for_stmt.stop.name == "n"
        assert len(for_stmt.iter_args) == 1

    def test_scalar_parallel_range(self):
        """Test pl.parallel(n) where n is a Scalar[INT64] parameter."""

        @pl.function
        def scalar_parallel(n: pl.Scalar[pl.INT64], x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            for i in pl.parallel(n):
                y: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
            return y

        assert isinstance(scalar_parallel, ir.Function)
        for_stmt = _find_for_stmt(scalar_parallel)
        assert isinstance(for_stmt.stop, ir.Var)
        assert for_stmt.stop.name == "n"
        assert for_stmt.kind == ir.ForKind.Parallel


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
