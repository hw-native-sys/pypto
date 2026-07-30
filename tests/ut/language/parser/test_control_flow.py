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
import pytest
from pypto import ir
from pypto.language.parser.diagnostics import InvalidOperationError, ParserSyntaxError


class TestForLoops:
    """Tests for for loop parsing."""

    def test_simple_for_loop(self):
        """Test simple for loop with one iter_arg."""

        @pl.function
        def sum_loop(n: pl.Tensor[[1], pl.INT32]) -> pl.Tensor[[1], pl.INT32]:
            init: pl.Tensor[[1], pl.INT32] = pl.create_tensor([1], dtype=pl.INT32)

            for i, (sum_val,) in pl.range(10, init_values=(init,)):
                # ``i`` is an INDEX loop var; cast before use as a scalar operand.
                new_sum: pl.Tensor[[1], pl.INT32] = pl.add(sum_val, pl.cast(i, pl.INT32))
                result = pl.yield_(new_sum)

            return result

        assert isinstance(sum_loop, ir.Function)
        assert sum_loop.name == "sum_loop"

    def test_for_loop_multiple_iter_args(self):
        """Test for loop with multiple iteration arguments."""

        @pl.function
        def multi_iter(n: pl.Tensor[[1], pl.INT32]) -> pl.Tensor[[1], pl.INT32]:
            init1: pl.Tensor[[1], pl.INT32] = pl.create_tensor([1], dtype=pl.INT32)
            init2: pl.Tensor[[1], pl.INT32] = pl.create_tensor([1], dtype=pl.INT32)

            for i, (val1, val2) in pl.range(5, init_values=(init1, init2)):
                new1: pl.Tensor[[1], pl.INT32] = pl.add(val1, pl.cast(i, pl.INT32))
                new2: pl.Tensor[[1], pl.INT32] = pl.mul(val2, 2)
                out1, out2 = pl.yield_(new1, new2)

            return out1

        for_stmt = _find_for_stmt(multi_iter)
        _assert_range(for_stmt, start=0, stop=5, step=1)
        assert len(for_stmt.iter_args) == 2
        assert len(for_stmt.return_vars) == 2
        _assert_body_kinds(for_stmt.body, ir.AssignStmt, ir.AssignStmt, ir.YieldStmt)

    def test_for_loop_with_range_params(self):
        """Test for loop with start, stop, step parameters."""

        @pl.function
        def range_params(n: pl.Tensor[[1], pl.INT32]) -> pl.Tensor[[1], pl.INT32]:
            init: pl.Tensor[[1], pl.INT32] = pl.create_tensor([1], dtype=pl.INT32)

            for i, (acc,) in pl.range(0, 10, 2, init_values=(init,)):
                new_acc: pl.Tensor[[1], pl.INT32] = pl.add(acc, pl.cast(i, pl.INT32))
                result = pl.yield_(new_acc)

            return result

        for_stmt = _find_for_stmt(range_params)
        _assert_range(for_stmt, start=0, stop=10, step=2)
        assert len(for_stmt.iter_args) == 1

    def test_nested_for_loops(self):
        """Test nested for loops."""

        @pl.function
        def nested_loops(n: pl.Tensor[[1], pl.INT32]) -> pl.Tensor[[1], pl.INT32]:
            init: pl.Tensor[[1], pl.INT32] = pl.create_tensor([1], dtype=pl.INT32)

            for i, (outer,) in pl.range(3, init_values=(init,)):
                for j, (inner,) in pl.range(2, init_values=(outer,)):
                    new_inner: pl.Tensor[[1], pl.INT32] = pl.add(inner, 1)
                    inner_out = pl.yield_(new_inner)

                outer_out = pl.yield_(inner_out)

            return outer_out

        outer = _find_for_stmt(nested_loops)
        _assert_range(outer, start=0, stop=3, step=1)
        _assert_body_kinds(outer.body, ir.ForStmt, ir.YieldStmt)

        inner = _top_level_stmts(outer.body)[0]
        assert isinstance(inner, ir.ForStmt), "outer loop body must open with the nested loop"
        _assert_range(inner, start=0, stop=2, step=1)
        assert len(inner.iter_args) == 1

    def test_for_loop_with_operations(self):
        """Test for loop with tensor operations."""

        @pl.function
        def loop_ops(x: pl.Tensor[[64, 128], pl.FP32]) -> pl.Tensor[[64, 128], pl.FP32]:
            init: pl.Tensor[[64, 128], pl.FP32] = pl.create_tensor([64, 128], dtype=pl.FP32)

            for i, (acc,) in pl.range(4, init_values=(init,)):
                temp: pl.Tensor[[64, 128], pl.FP32] = pl.add(acc, x)
                result = pl.yield_(temp)

            return result

        for_stmt = _find_for_stmt(loop_ops)
        _assert_range(for_stmt, start=0, stop=4, step=1)
        assert len(for_stmt.iter_args) == 1
        _assert_body_kinds(for_stmt.body, ir.AssignStmt, ir.YieldStmt)


class TestIfStatements:
    """Tests for if statement parsing.

    Note: If conditions currently require scalar types, not tensors.
    Tests use scalar loop variables for conditions.
    """

    def test_if_in_loop(self):
        """Test if statement inside for loop."""

        @pl.function
        def if_in_loop(n: pl.Tensor[[1], pl.INT32]) -> pl.Tensor[[64], pl.FP32]:
            init: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)

            for i, (acc,) in pl.range(5, init_values=(init,)):
                if i == 0:
                    new_val: pl.Tensor[[64], pl.FP32] = pl.mul(acc, 2.0)
                    val: pl.Tensor[[64], pl.FP32] = pl.yield_(new_val)
                else:
                    val: pl.Tensor[[64], pl.FP32] = pl.yield_(acc)

                result = pl.yield_(val)

            return result

        for_stmt = _find_for_stmt(if_in_loop)
        _assert_range(for_stmt, start=0, stop=5, step=1)
        _assert_body_kinds(for_stmt.body, ir.IfStmt, ir.YieldStmt)

        if_stmt = _top_level_stmts(for_stmt.body)[0]
        assert isinstance(if_stmt, ir.IfStmt)
        assert isinstance(if_stmt.condition, ir.Eq), "condition `i == 0` must parse to Eq"
        assert if_stmt.else_body is not None, "else branch must be preserved"
        assert len(if_stmt.return_vars) == 1

    def test_if_yield_type_annotation_preserved(self):
        """Test that type annotations on yield assignments are preserved in IR (issue #185)."""

        @pl.function
        def if_with_annotated_yield(n: pl.Tensor[[1], pl.INT32]) -> pl.Tensor[[64, 128], pl.FP32]:
            init: pl.Tensor[[64, 128], pl.FP32] = pl.create_tensor([64, 128], dtype=pl.FP32)

            for i, (acc,) in pl.range(5, init_values=(init,)):
                if i == 0:
                    out_c: pl.Tensor[[64, 128], pl.FP32] = pl.mul(acc, 2.0)
                    val: pl.Tensor[[64, 128], pl.FP32] = pl.yield_(out_c)
                else:
                    val: pl.Tensor[[64, 128], pl.FP32] = pl.yield_(acc)

                result = pl.yield_(val)

            return result

        # Verify function was created successfully
        assert isinstance(if_with_annotated_yield, ir.Function)

        # Verify that the type annotation was preserved in the IR by printing and parsing
        printed = if_with_annotated_yield.as_python()
        # The printed output should contain the type annotation on the yield
        assert "val: pl.Tensor[[64, 128], pl.FP32] = pl.yield_" in printed

    def test_if_yield_type_inferred_from_expression(self):
        """Test that unannotated yield assignments infer the type from the yielded expression (issue #234)."""

        @pl.function
        def if_with_unannotated_yield(
            x: pl.Tensor[[64], pl.FP32], n: pl.Scalar[pl.INT64]
        ) -> pl.Tensor[[64], pl.FP32]:
            if n == 0:
                y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                z = pl.yield_(y)  # No annotation - should infer Tensor[[64], FP32]
            else:
                y2: pl.Tensor[[64], pl.FP32] = pl.mul(x, x)
                z = pl.yield_(y2)
            return z

        assert isinstance(if_with_unannotated_yield, ir.Function)
        printed = if_with_unannotated_yield.as_python()
        # Should infer correct type, not Tensor[[1], INT32]
        assert "Tensor[[1], pl.INT32]" not in printed
        assert "Tensor[[64], pl.FP32]" in printed


class TestComplexControlFlow:
    """Tests for complex control flow combinations."""

    def test_loop_with_if_and_multiple_iter_args(self):
        """Test loop with if statement and multiple iter_args."""

        @pl.function
        def complex_flow(n: pl.Tensor[[1], pl.INT32]) -> pl.Tensor[[64], pl.FP32]:
            acc1: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
            acc2: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)

            for i, (a1, a2) in pl.range(10, init_values=(acc1, acc2)):
                if i == 0:
                    new1: pl.Tensor[[64], pl.FP32] = pl.mul(a1, 2.0)
                    new2: pl.Tensor[[64], pl.FP32] = pl.mul(a2, 3.0)
                    val1, val2 = pl.yield_(new1, new2)
                else:
                    val1, val2 = pl.yield_(a1, a2)

                out1, out2 = pl.yield_(val1, val2)

            return out1

        for_stmt = _find_for_stmt(complex_flow)
        _assert_range(for_stmt, start=0, stop=10, step=1)
        assert len(for_stmt.iter_args) == 2
        assert len(for_stmt.return_vars) == 2
        _assert_body_kinds(for_stmt.body, ir.IfStmt, ir.YieldStmt)

        if_stmt = _top_level_stmts(for_stmt.body)[0]
        assert isinstance(if_stmt, ir.IfStmt)
        assert if_stmt.else_body is not None
        assert len(if_stmt.return_vars) == 2, "both branches yield two values"

    def test_sequential_loops(self):
        """Test sequential (not nested) for loops."""

        @pl.function
        def sequential_loops(n: pl.Tensor[[1], pl.INT32]) -> pl.Tensor[[64], pl.FP32]:
            init: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)

            # First loop
            for i, (acc,) in pl.range(5, init_values=(init,)):
                new_acc: pl.Tensor[[64], pl.FP32] = pl.add(acc, 1.0)
                result1 = pl.yield_(new_acc)

            # Second loop uses output of first
            for j, (acc2,) in pl.range(3, init_values=(result1,)):
                new_acc2: pl.Tensor[[64], pl.FP32] = pl.mul(acc2, 2.0)
                result2 = pl.yield_(new_acc2)

            return result2

        loops = _find_all_for_stmts(sequential_loops)
        assert len(loops) == 2, "the two loops must stay siblings, not nest"
        _assert_range(loops[0], start=0, stop=5, step=1)
        _assert_range(loops[1], start=0, stop=3, step=1)
        # Second loop consumes the first loop's result as its init value.
        assert loops[1].iter_args[0].name_hint == "acc2"

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

        for_stmt = _find_for_stmt(loop_without_iter_args)
        _assert_range(for_stmt, start=0, stop=3, step=1)
        assert len(for_stmt.iter_args) == 0
        assert len(for_stmt.return_vars) == 0
        _assert_body_kinds(for_stmt.body, ir.IfStmt)


class TestParallelForLoops:
    """Tests for parallel for loop parsing with pl.parallel()."""

    def test_simple_parallel_for_loop(self):
        """Test simple parallel for loop with one iter_arg."""

        @pl.function
        def parallel_sum(n: pl.Tensor[[1], pl.INT32]) -> pl.Tensor[[1], pl.INT32]:
            init: pl.Tensor[[1], pl.INT32] = pl.create_tensor([1], dtype=pl.INT32)

            for i, (sum_val,) in pl.parallel(10, init_values=(init,)):
                new_sum: pl.Tensor[[1], pl.INT32] = pl.add(sum_val, pl.cast(i, pl.INT32))
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

        for_stmt = _find_for_stmt(parallel_no_iter)
        assert for_stmt.kind == ir.ForKind.Parallel
        _assert_range(for_stmt, start=0, stop=3, step=1)
        assert len(for_stmt.iter_args) == 0
        assert len(for_stmt.return_vars) == 0

    def test_parallel_for_loop_with_range_params(self):
        """Test parallel for loop with start, stop, step parameters."""

        @pl.function
        def parallel_range(n: pl.Tensor[[1], pl.INT32]) -> pl.Tensor[[1], pl.INT32]:
            init: pl.Tensor[[1], pl.INT32] = pl.create_tensor([1], dtype=pl.INT32)

            for i, (acc,) in pl.parallel(0, 10, 2, init_values=(init,)):
                new_acc: pl.Tensor[[1], pl.INT32] = pl.add(acc, pl.cast(i, pl.INT32))
                result = pl.yield_(new_acc)

            return result

        for_stmt = _find_for_stmt(parallel_range)
        assert for_stmt.kind == ir.ForKind.Parallel
        _assert_range(for_stmt, start=0, stop=10, step=2)
        assert len(for_stmt.iter_args) == 1

    def test_parallel_for_produces_parallel_kind(self):
        """Test that pl.parallel() produces ForKind.Parallel in the IR."""

        @pl.function
        def par_func(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            init: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
            for i, (acc,) in pl.parallel(10, init_values=(init,)):
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
            init: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
            for i, (acc,) in pl.range(10, init_values=(init,)):
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
                init: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                for i, (acc,) in pl.parallel(10, init_values=(init,)):
                    new_acc: pl.Tensor[[64], pl.FP32] = pl.add(acc, x)
                    result = pl.yield_(new_acc)
                return result

        printed = Before.as_python()
        assert "pl.parallel(" in printed
        assert "pl.range(" not in printed

    def test_sequential_for_printer_no_parallel(self):
        """Test that sequential for loop does not print pl.parallel()."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                init: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                for i, (acc,) in pl.range(10, init_values=(init,)):
                    new_acc: pl.Tensor[[64], pl.FP32] = pl.add(acc, x)
                    result = pl.yield_(new_acc)
                return result

        printed = Before.as_python()
        assert "pl.parallel(" not in printed
        assert "pl.range(" in printed

    def test_parallel_for_structural_not_equal_to_sequential(self):
        """Test that parallel and sequential loops with same body are not structurally equal."""

        @pl.program
        class ParallelProg:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                init: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                for i, (acc,) in pl.parallel(10, init_values=(init,)):
                    new_acc: pl.Tensor[[64], pl.FP32] = pl.add(acc, x)
                    result = pl.yield_(new_acc)
                return result

        @pl.program
        class SequentialProg:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                init: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                for i, (acc,) in pl.range(10, init_values=(init,)):
                    new_acc: pl.Tensor[[64], pl.FP32] = pl.add(acc, x)
                    result = pl.yield_(new_acc)
                return result

        assert not ir.structural_equal(ParallelProg, SequentialProg)

    def test_invalid_iterator_rejected(self):
        """Test that invalid iterator (not range or parallel) is rejected."""
        with pytest.raises(ParserSyntaxError):

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


def _top_level_stmts(stmt: ir.Stmt) -> list[ir.Stmt]:
    """Return the statement list of a body, treating a bare Stmt as a 1-element body."""
    return list(stmt.stmts) if isinstance(stmt, ir.SeqStmts) else [stmt]


def _find_all_for_stmts(func: ir.Function) -> list[ir.ForStmt]:
    """Helper to extract every top-level ForStmt from a function body, in order."""
    return [s for s in _top_level_stmts(func.body) if isinstance(s, ir.ForStmt)]


def _assert_range(for_stmt: ir.ForStmt, start: int, stop: int, step: int) -> None:
    """Assert a ForStmt's constant loop bounds.

    Pins the parsed ``pl.range(...)`` arguments; ``isinstance(stmt, ir.ForStmt)``
    alone does not distinguish ``range(0, 10, 2)`` from ``range(10)``.
    """
    for name, expected in (("start", start), ("stop", stop), ("step", step)):
        bound = getattr(for_stmt, name)
        assert isinstance(bound, ir.ConstInt), f"{name} is {type(bound).__name__}, expected a constant"
        assert bound.value == expected, f"{name} is {bound.value}, expected {expected}"


def _assert_var_named(expr: ir.Expr, name: str) -> None:
    """Assert `expr` is a Var reference bound to `name`."""
    assert isinstance(expr, ir.Var), f"expected Var {name!r}, got {type(expr).__name__}"
    assert expr.name_hint == name, f"expected Var {name!r}, got {expr.name_hint!r}"


def _assert_body_kinds(body: ir.Stmt, *kinds: type) -> None:
    """Assert the exact statement kinds making up a loop/branch body, in order."""
    actual = [type(s) for s in _top_level_stmts(body)]
    assert actual == list(kinds), (
        f"body is {[k.__name__ for k in actual]}, expected {[k.__name__ for k in kinds]}"
    )


def _find_while_stmt(func: ir.Function) -> ir.WhileStmt:
    """Helper to find WhileStmt in function body."""
    stmt = func.body
    # Handle SeqStmts wrapper
    if isinstance(stmt, ir.SeqStmts):
        for s in stmt.stmts:
            if isinstance(s, ir.WhileStmt):
                return s
            # Check nested statements
            if isinstance(s, ir.ForStmt) and isinstance(s.body, ir.SeqStmts):
                for nested in s.body.stmts:
                    if isinstance(nested, ir.WhileStmt):
                        return nested
    # Direct while statement
    if isinstance(stmt, ir.WhileStmt):
        return stmt
    raise ValueError("No WhileStmt found in function body")


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
        assert for_stmt.stop.name_hint == "n"
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
        assert for_stmt.stop.name_hint == "n"

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
        assert for_stmt.stop.name_hint == "n"
        assert isinstance(for_stmt.step, ir.Var)
        assert for_stmt.step.name_hint == "s"

    def test_scalar_expression_as_stop(self):
        """Test pl.range(n * 2) where n is a Scalar[INT64] parameter."""

        @pl.function
        def scalar_expr_stop(n: pl.Scalar[pl.INT64], x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            for i in pl.range(n * 2):
                y: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
            return y

        for_stmt = _find_for_stmt(scalar_expr_stop)
        # stop should be exactly `n * 2`, not merely "some Mul"
        assert isinstance(for_stmt.stop, ir.Mul)
        assert for_stmt.stop.left is scalar_expr_stop.params[0]
        assert isinstance(for_stmt.stop.right, ir.ConstInt)
        assert for_stmt.stop.right.value == 2

    def test_scalar_complex_expression_as_stop(self):
        """Test pl.range(n * 2 + 1) where n is a Scalar[INT64] parameter."""

        @pl.function
        def scalar_complex_expr(
            n: pl.Scalar[pl.INT64], x: pl.Tensor[[64], pl.FP32]
        ) -> pl.Tensor[[64], pl.FP32]:
            for i in pl.range(n * 2 + 1):
                y: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
            return y

        for_stmt = _find_for_stmt(scalar_complex_expr)
        # stop should be exactly `(n * 2) + 1`, preserving precedence
        stop = for_stmt.stop
        assert isinstance(stop, ir.Add)
        assert isinstance(stop.right, ir.ConstInt)
        assert stop.right.value == 1
        assert isinstance(stop.left, ir.Mul)
        assert stop.left.left is scalar_complex_expr.params[0]
        assert isinstance(stop.left.right, ir.ConstInt)
        assert stop.left.right.value == 2

    def test_scalar_floordiv_expression_as_stop(self):
        """Test pl.range(n // 4) where n is a Scalar[INT64] parameter."""

        @pl.function
        def scalar_floordiv_expr(
            n: pl.Scalar[pl.INT64], x: pl.Tensor[[64], pl.FP32]
        ) -> pl.Tensor[[64], pl.FP32]:
            for i in pl.range(n // 4):
                y: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
            return y

        for_stmt = _find_for_stmt(scalar_floordiv_expr)
        # stop should be exactly `n // 4`
        assert isinstance(for_stmt.stop, ir.FloorDiv)
        assert for_stmt.stop.left is scalar_floordiv_expr.params[0]
        assert isinstance(for_stmt.stop.right, ir.ConstInt)
        assert for_stmt.stop.right.value == 4

    def test_scalar_range_with_iter_args(self):
        """Test pl.range(n, init_values=(init,)) where n is a Scalar[INT64] parameter."""

        @pl.function
        def scalar_range_iter(
            n: pl.Scalar[pl.INT64], x: pl.Tensor[[64], pl.FP32]
        ) -> pl.Tensor[[64], pl.FP32]:
            init: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
            for i, (acc,) in pl.range(n, init_values=(init,)):
                new_acc: pl.Tensor[[64], pl.FP32] = pl.add(acc, x)
                result = pl.yield_(new_acc)
            return result

        assert isinstance(scalar_range_iter, ir.Function)
        for_stmt = _find_for_stmt(scalar_range_iter)
        assert isinstance(for_stmt.stop, ir.Var)
        assert for_stmt.stop.name_hint == "n"
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
        assert for_stmt.stop.name_hint == "n"
        assert for_stmt.kind == ir.ForKind.Parallel


class TestWhileLoops:
    """Tests for while loop parsing."""

    def test_natural_while_loop(self):
        """Test natural while loop syntax (non-SSA form)."""

        @pl.function
        def natural_while(n: pl.Scalar[pl.INT64]) -> pl.Scalar[pl.INT64]:
            x: pl.Scalar[pl.INT64] = 0
            while x < n:
                x = x + 1
            return x

        assert isinstance(natural_while, ir.Function)
        assert natural_while.name == "natural_while"

        # Find the while statement
        while_stmt = _find_while_stmt(natural_while)
        assert isinstance(while_stmt, ir.WhileStmt)

        # Natural syntax has no iter_args initially (ConvertToSSA adds them later)
        assert len(while_stmt.iter_args) == 0
        assert len(while_stmt.return_vars) == 0

        # Condition should be a comparison
        assert isinstance(while_stmt.condition, ir.Lt)

        # Body should be present
        assert while_stmt.body is not None

    def test_natural_while_loop_with_initialization(self):
        """Test natural while loop with explicit initialization."""

        @pl.function
        def natural_while_init(limit: pl.Scalar[pl.INT64]) -> pl.Scalar[pl.INT64]:
            counter: pl.Scalar[pl.INT64] = 0
            sum_val: pl.Scalar[pl.INT64] = 0
            while counter < limit:
                sum_val = sum_val + counter
                counter = counter + 1
            return sum_val

        assert isinstance(natural_while_init, ir.Function)

        # Find the while statement
        while_stmt = _find_while_stmt(natural_while_init)
        assert isinstance(while_stmt, ir.WhileStmt)

        # Check condition references 'counter' and 'limit'
        assert isinstance(while_stmt.condition, ir.Lt)

        # Natural form has no SSA iter_args
        assert len(while_stmt.iter_args) == 0

    def test_while_loop_with_tensors(self):
        """Test while loop with tensor operations."""

        @pl.function
        def while_tensors(n: pl.Scalar[pl.INT64], x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            i: pl.Scalar[pl.INT64] = 0
            acc: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
            while i < n:
                i = i + 1
                acc = pl.add(acc, x)
            return acc

        while_stmt = _find_while_stmt(while_tensors)

        # Condition is `i < n`, with `n` the Scalar parameter
        assert isinstance(while_stmt.condition, ir.Lt)
        _assert_var_named(while_stmt.condition.left, "i")
        assert while_stmt.condition.right is while_tensors.params[0]

        # Body holds both assignments: the counter bump and the tensor accumulate
        _assert_body_kinds(while_stmt.body, ir.AssignStmt, ir.AssignStmt)

    def test_nested_while_loops(self):
        """Test nested while loops."""

        @pl.function
        def nested_while(n: pl.Scalar[pl.INT64]) -> pl.Scalar[pl.INT64]:
            x: pl.Scalar[pl.INT64] = 0
            while x < n:
                y: pl.Scalar[pl.INT64] = 0
                while y < 3:
                    y = y + 1
                x = x + 1
            return x

        outer_while = _find_while_stmt(nested_while)

        # Outer condition is `x < n`, with `n` the Scalar parameter
        assert isinstance(outer_while.condition, ir.Lt)
        _assert_var_named(outer_while.condition.left, "x")
        assert outer_while.condition.right is nested_while.params[0]

        # Outer body: `y = 0`, the nested loop, then `x = x + 1`
        _assert_body_kinds(outer_while.body, ir.AssignStmt, ir.WhileStmt, ir.AssignStmt)

        inner_while = _top_level_stmts(outer_while.body)[1]
        assert isinstance(inner_while, ir.WhileStmt)

        # Inner condition is exactly `y < 3`
        assert isinstance(inner_while.condition, ir.Lt)
        _assert_var_named(inner_while.condition.left, "y")
        assert isinstance(inner_while.condition.right, ir.ConstInt)
        assert inner_while.condition.right.value == 3
        _assert_body_kinds(inner_while.body, ir.AssignStmt)

    def test_while_inside_for_loop(self):
        """Test while loop nested inside a for loop."""

        @pl.function
        def while_in_for(n: pl.Scalar[pl.INT64]) -> pl.Scalar[pl.INT64]:
            init_sum: pl.Scalar[pl.INT64] = 0

            for i, (sum_val,) in pl.range(5, init_values=(init_sum,)):
                x: pl.Scalar[pl.INT64] = 0
                while x < i:
                    x = x + 1
                new_sum: pl.Scalar[pl.INT64] = sum_val + x
                sum_out = pl.yield_(new_sum)

            return sum_out

        for_stmt = _find_for_stmt(while_in_for)
        _assert_range(for_stmt, start=0, stop=5, step=1)
        assert len(for_stmt.iter_args) == 1

        # For body: `x = 0`, the while loop, `new_sum = ...`, then the yield
        _assert_body_kinds(for_stmt.body, ir.AssignStmt, ir.WhileStmt, ir.AssignStmt, ir.YieldStmt)

        while_stmt = _top_level_stmts(for_stmt.body)[1]
        assert isinstance(while_stmt, ir.WhileStmt)

        # The while condition closes over the enclosing loop variable: `x < i`
        assert isinstance(while_stmt.condition, ir.Lt)
        _assert_var_named(while_stmt.condition.left, "x")
        assert while_stmt.condition.right is for_stmt.loop_var
        _assert_body_kinds(while_stmt.body, ir.AssignStmt)

    def test_for_inside_while_loop(self):
        """Test for loop nested inside a while loop."""

        @pl.function
        def for_in_while(n: pl.Scalar[pl.INT64]) -> pl.Scalar[pl.INT64]:
            x: pl.Scalar[pl.INT64] = 0

            while x < n:
                init_acc: pl.Scalar[pl.INT64] = x
                for i, (acc,) in pl.range(3, init_values=(init_acc,)):
                    new_acc: pl.Scalar[pl.INT64] = acc + 1
                    acc_out = pl.yield_(new_acc)
                x = acc_out

            return x

        assert isinstance(for_in_while, ir.Function)

        # Find the while statement
        while_stmt = _find_while_stmt(for_in_while)
        assert isinstance(while_stmt, ir.WhileStmt)

        # Find for statement inside the while loop body
        for_stmt = None
        if isinstance(while_stmt.body, ir.SeqStmts):
            for stmt in while_stmt.body.stmts:
                if isinstance(stmt, ir.ForStmt):
                    for_stmt = stmt
                    break

        assert for_stmt is not None, "Expected ForStmt in while loop body"
        assert isinstance(for_stmt, ir.ForStmt)
        assert for_stmt.kind == ir.ForKind.Sequential

    def test_while_with_multiple_updates(self):
        """Test while loop with multiple variable updates."""

        @pl.function
        def while_multi_update(n: pl.Scalar[pl.INT64]) -> pl.Scalar[pl.INT64]:
            x: pl.Scalar[pl.INT64] = 0
            y: pl.Scalar[pl.INT64] = 1

            while x < n:
                x = x + 1
                y = y * 2

            return y

        assert isinstance(while_multi_update, ir.Function)

        # Find the while statement
        while_stmt = _find_while_stmt(while_multi_update)
        assert isinstance(while_stmt, ir.WhileStmt)

        # Body should contain multiple assignments
        assert while_stmt.body is not None
        if isinstance(while_stmt.body, ir.SeqStmts):
            # Should have at least 2 statements (x = x + 1, y = y * 2)
            assert len(while_stmt.body.stmts) >= 2


class TestBreakContinue:
    """Tests for break and continue statement parsing with roundtrip verification."""

    def test_break_in_sequential_for_loop(self):
        """Test break in a sequential for loop with roundtrip."""

        @pl.program
        class Before:
            @pl.function
            def main(self, n: pl.Scalar[pl.INT64]) -> pl.Scalar[pl.INT64]:
                x: pl.Scalar[pl.INT64] = 0
                for i in pl.range(10):
                    x = x + 1
                    break
                return x

        printed = Before.as_python()
        assert "break" in printed
        reparsed = pl.parse_program(printed)
        ir.assert_structural_equal(Before, reparsed)

    def test_continue_in_sequential_for_loop(self):
        """Test continue in a sequential for loop with roundtrip."""

        @pl.program
        class Before:
            @pl.function
            def main(self, n: pl.Scalar[pl.INT64]) -> pl.Scalar[pl.INT64]:
                x: pl.Scalar[pl.INT64] = 0
                for i in pl.range(10):
                    continue
                return x

        printed = Before.as_python()
        assert "continue" in printed
        reparsed = pl.parse_program(printed)
        ir.assert_structural_equal(Before, reparsed)

    def test_break_in_while_loop(self):
        """Test break in a while loop with roundtrip."""

        @pl.program
        class Before:
            @pl.function
            def main(self, n: pl.Scalar[pl.INT64]) -> pl.Scalar[pl.INT64]:
                x: pl.Scalar[pl.INT64] = 0
                while x < n:
                    x = x + 1
                    break
                return x

        printed = Before.as_python()
        assert "break" in printed
        reparsed = pl.parse_program(printed)
        ir.assert_structural_equal(Before, reparsed)

    def test_continue_in_while_loop(self):
        """Test continue in a while loop with roundtrip."""

        @pl.program
        class Before:
            @pl.function
            def main(self, n: pl.Scalar[pl.INT64]) -> pl.Scalar[pl.INT64]:
                x: pl.Scalar[pl.INT64] = 0
                while x < n:
                    x = x + 1
                    continue
                return x

        printed = Before.as_python()
        assert "continue" in printed
        reparsed = pl.parse_program(printed)
        ir.assert_structural_equal(Before, reparsed)

    def test_break_in_nested_loops(self):
        """Test break in inner sequential loop inside outer loop with roundtrip."""

        @pl.program
        class Before:
            @pl.function
            def main(self, n: pl.Scalar[pl.INT64]) -> pl.Scalar[pl.INT64]:
                x: pl.Scalar[pl.INT64] = 0
                for i in pl.range(5):
                    for j in pl.range(3):
                        break
                    x = x + 1
                return x

        printed = Before.as_python()
        assert "break" in printed
        reparsed = pl.parse_program(printed)
        ir.assert_structural_equal(Before, reparsed)

    def test_break_and_continue_in_same_loop(self):
        """Test both break and continue in the same loop with roundtrip."""

        @pl.program
        class Before:
            @pl.function
            def main(self, n: pl.Scalar[pl.INT64]) -> pl.Scalar[pl.INT64]:
                x: pl.Scalar[pl.INT64] = 0
                for i in pl.range(10):
                    if i == 5:
                        break
                    else:
                        continue
                return x

        printed = Before.as_python()
        assert "break" in printed
        assert "continue" in printed
        reparsed = pl.parse_program(printed)
        ir.assert_structural_equal(Before, reparsed)

    def test_break_in_seq_inside_parallel(self):
        """Test break in inner sequential loop inside outer parallel loop with roundtrip."""

        @pl.program
        class Before:
            @pl.function
            def main(self, n: pl.Scalar[pl.INT64]) -> pl.Scalar[pl.INT64]:
                x: pl.Scalar[pl.INT64] = 0
                for i in pl.parallel(5):
                    for j in pl.range(3):
                        break
                    x = x + 1
                return x

        printed = Before.as_python()
        assert "break" in printed
        assert "pl.parallel(" in printed
        reparsed = pl.parse_program(printed)
        ir.assert_structural_equal(Before, reparsed)


class TestBreakContinueErrors:
    """Tests for break and continue error cases."""

    def test_break_in_parallel_loop(self):
        """Test that break in parallel loop raises InvalidOperationError."""
        with pytest.raises(InvalidOperationError, match="parallel"):

            @pl.function
            def bad_parallel_break(n: pl.Scalar[pl.INT64]) -> pl.Scalar[pl.INT64]:
                x: pl.Scalar[pl.INT64] = 0
                for i in pl.parallel(10):
                    break
                return x

    def test_continue_in_parallel_loop(self):
        """Test that continue in parallel loop raises InvalidOperationError."""
        with pytest.raises(InvalidOperationError, match="parallel"):

            @pl.function
            def bad_parallel_continue(n: pl.Scalar[pl.INT64]) -> pl.Scalar[pl.INT64]:
                x: pl.Scalar[pl.INT64] = 0
                for i in pl.parallel(10):
                    continue
                return x

    def test_break_in_unroll_loop(self):
        """Test that break in unrolled loop raises InvalidOperationError."""
        with pytest.raises(InvalidOperationError, match="unrolled"):

            @pl.function
            def bad_unroll_break(n: pl.Scalar[pl.INT64]) -> pl.Scalar[pl.INT64]:
                x: pl.Scalar[pl.INT64] = 0
                for i in pl.unroll(4):
                    break
                return x

    def test_continue_in_unroll_loop(self):
        """Test that continue in unrolled loop raises InvalidOperationError."""
        with pytest.raises(InvalidOperationError, match="unrolled"):

            @pl.function
            def bad_unroll_continue(n: pl.Scalar[pl.INT64]) -> pl.Scalar[pl.INT64]:
                x: pl.Scalar[pl.INT64] = 0
                for i in pl.unroll(4):
                    continue
                return x

    # Note: "break/continue outside loop" is caught by Python's own syntax checker
    # before the DSL parser runs, so it cannot be tested via @pl.function.
    # The C++ BreakContinueCheck verifier covers this at the IR level.


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
