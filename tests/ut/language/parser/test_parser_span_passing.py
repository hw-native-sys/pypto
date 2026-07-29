# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for parser passing span information to operations."""

import inspect

import pypto.language as pl
import pytest
from pypto import ir


def get_current_line():
    """Get the current line number in the calling code."""
    frame = inspect.currentframe()
    if frame and frame.f_back:
        return frame.f_back.f_lineno
    return -1


def top_level_calls(func: ir.Function) -> list[ir.Call]:
    """Return every ``AssignStmt``-bound ``Call`` in ``func``'s body, in source order.

    Only the top level of the body is scanned -- these tests assert on the span of the
    call the surface syntax binds directly, not on nested argument expressions.

    Args:
        func: The parsed function to scan.

    Returns:
        The bound ``Call`` expressions, in statement order.
    """
    body = func.body
    assert isinstance(body, ir.SeqStmts)
    return [
        stmt.value
        for stmt in body.stmts
        if isinstance(stmt, ir.AssignStmt) and isinstance(stmt.value, ir.Call)
    ]


def find_unique_call(func: ir.Function, op_name: str) -> ir.Call:
    """Return the single ``AssignStmt``-bound ``Call`` in ``func`` carrying ``op_name``.

    Routing ``op_name`` through ``ir.get_op`` makes a stale operator name raise
    instead of silently evaluating to ``False``. Asserting the match was found
    is what keeps the span checks at the call site from being skipped outright
    if the parser stops emitting ``op_name`` for the tested surface syntax --
    without it, a dead search reports PASS while verifying nothing.

    Args:
        func: The parsed function to search.
        op_name: Registered operator name the call is expected to carry.

    Returns:
        The matching ``Call`` expression.

    Raises:
        ValueError: If ``op_name`` is not a registered operator.
    """
    want = ir.get_op(op_name).name
    calls = top_level_calls(func)
    hits = [call for call in calls if call.op.name == want]

    assert len(hits) == 1, (
        f"Expected exactly one {op_name!r} call in {func.name!r}, got {len(hits)}; "
        f"body contained {[call.op.name for call in calls]}"
    )
    return hits[0]


def assert_span_at(
    span: ir.Span, expected_line: int, expected_column: int | None = None, context: str = ""
) -> None:
    """Assert ``span`` is a valid, well-ordered range beginning at ``expected_line``.

    Args:
        span: The span to check.
        expected_line: Line number the span must begin on.
        expected_column: Exact begin column, or None to only require it is positive.
        context: Optional label identifying the span, used in failure messages.
    """
    where = f" for {context}" if context else ""
    assert span.is_valid(), f"Invalid span{where}"
    assert span.begin_line == expected_line, (
        f"Invalid begin line{where}: {span.begin_line} != {expected_line}"
    )
    if expected_column is None:
        assert span.begin_column > 0, f"Invalid begin column{where}: {span.begin_column}"
    else:
        assert span.begin_column == expected_column, (
            f"Invalid begin column{where}: {span.begin_column} != {expected_column}"
        )
    # Equivalent to "end line is later, or same line with a non-decreasing column",
    # but stated unconditionally so the ordering check can never be skipped.
    assert (span.end_line, span.end_column) >= (span.begin_line, span.begin_column), (
        f"Span end precedes begin{where}: "
        f"({span.end_line}, {span.end_column}) < ({span.begin_line}, {span.begin_column})"
    )


class TestParserSpanPassing:
    """Test that parser passes accurate span information to operations."""

    def test_parser_passes_span_to_tensor_add(self):
        """Parser should pass AST span to tensor.add operation."""

        current_line = get_current_line()

        @pl.function
        def test_func(
            x: pl.Tensor[[64], pl.FP32],
            y: pl.Tensor[[64], pl.FP32],
        ) -> pl.Tensor[[64], pl.FP32]:
            z: pl.Tensor[[64], pl.FP32] = pl.add(x, y)  # Line current_line + 7
            return z

        # Function should be created successfully
        assert isinstance(test_func, ir.Function)
        assert test_func.name == "test_func"

        add_call = find_unique_call(test_func, "tensor.add")
        assert_span_at(add_call.span, current_line + 7)

    def test_parser_passes_span_to_tensor_mul(self):
        """Parser should pass AST span to the tensor.muls operation a scalar rhs lowers to."""

        current_line = get_current_line()

        @pl.function
        def test_mul(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            y: pl.Tensor[[64], pl.FP32] = pl.mul(x, 2.0)  # Line current_line + 4
            return y

        assert isinstance(test_mul, ir.Function)

        # A scalar rhs lowers to tensor.muls, not tensor.mul
        mul_call = find_unique_call(test_mul, "tensor.muls")
        assert_span_at(mul_call.span, current_line + 4)

    def test_parser_passes_span_to_tensor_create(self):
        """Parser should pass AST span to tensor.create operation."""

        current_line = get_current_line()

        @pl.function
        def test_create() -> pl.Tensor[[64, 32], pl.FP32]:
            x: pl.Tensor[[64, 32], pl.FP32] = pl.create_tensor([64, 32], dtype=pl.FP32)  # current_line + 4
            return x

        assert isinstance(test_create, ir.Function)

        create_call = find_unique_call(test_create, "tensor.create")
        assert_span_at(create_call.span, current_line + 4, expected_column=46)

    def test_parser_span_accuracy_multiple_operations(self):
        """Test that parser assigns different spans to different operations."""

        current_line = get_current_line()

        @pl.function
        def test_multi(x: pl.Tensor[[32], pl.FP32]) -> pl.Tensor[[32], pl.FP32]:
            y: pl.Tensor[[32], pl.FP32] = pl.mul(x, 2.0)  # current_line + 4
            z: pl.Tensor[[32], pl.FP32] = pl.add(y, 1.0)  # current_line + 5
            return z

        assert isinstance(test_multi, ir.Function)

        # State the exact contract the zip below enforces, so an extra emitted call
        # fails as a named length assertion rather than a bare zip ValueError.
        calls = top_level_calls(test_multi)
        assert len(calls) == 2, f"expected 2 calls, got {[call.op.name for call in calls]}"

        for call, offset in zip(calls, [4, 5], strict=True):
            assert_span_at(call.span, current_line + offset, expected_column=42, context=call.op.name)

    def test_parser_passes_span_to_matmul(self):
        """Parser should pass AST span to tensor.matmul operation."""

        current_line = get_current_line()

        @pl.function
        def test_matmul(
            a: pl.Tensor[[64, 32], pl.FP32],
            b: pl.Tensor[[32, 16], pl.FP32],
        ) -> pl.Tensor[[64, 16], pl.FP32]:
            c: pl.Tensor[[64, 16], pl.FP32] = pl.matmul(a, b)  # current_line + 7
            return c

        assert isinstance(test_matmul, ir.Function)

        matmul_call = find_unique_call(test_matmul, "tensor.matmul")
        assert_span_at(matmul_call.span, current_line + 7)

    def test_parser_passes_span_to_cast(self):
        """Parser should pass AST span to tensor.cast operation."""

        current_line = get_current_line()

        @pl.function
        def test_cast(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP16]:
            y: pl.Tensor[[64], pl.FP16] = pl.cast(x, target_type=pl.FP16)  # current_line + 4
            return y

        assert isinstance(test_cast, ir.Function)

        cast_call = find_unique_call(test_cast, "tensor.cast")
        assert_span_at(cast_call.span, current_line + 4)

    def test_parser_passes_span_to_exp(self):
        """Parser should pass AST span to tensor.exp operation."""

        current_line = get_current_line()

        @pl.function
        def test_exp(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            y: pl.Tensor[[64], pl.FP32] = pl.exp(x)  # current_line + 4
            return y

        assert isinstance(test_exp, ir.Function)

        exp_call = find_unique_call(test_exp, "tensor.exp")
        assert_span_at(exp_call.span, current_line + 4)

    def test_all_operations_have_valid_spans(self):
        """Comprehensive test that all operations get valid spans from parser."""

        current_line = get_current_line()

        @pl.function
        def test_comprehensive(
            x: pl.Tensor[[64], pl.FP32],
            y: pl.Tensor[[64], pl.FP32],
        ) -> pl.Tensor[[64], pl.FP32]:
            a: pl.Tensor[[64], pl.FP32] = pl.add(x, y)  # current_line + 7
            b: pl.Tensor[[64], pl.FP32] = pl.sub(a, 1.0)  # current_line + 8
            c: pl.Tensor[[64], pl.FP32] = pl.mul(b, 2.0)  # current_line + 9
            d: pl.Tensor[[64], pl.FP32] = pl.div(c, 3.0)  # current_line + 10
            e: pl.Tensor[[64], pl.FP32] = pl.exp(d)  # current_line + 11
            return e

        assert isinstance(test_comprehensive, ir.Function)

        calls = top_level_calls(test_comprehensive)
        assert len(calls) == 5, f"expected 5 calls, got {[call.op.name for call in calls]}"

        for call, offset in zip(calls, [7, 8, 9, 10, 11], strict=True):
            assert_span_at(call.span, current_line + offset, context=call.op.name)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
