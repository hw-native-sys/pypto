# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Tests for tuple literal and subscript syntax in parser."""

from collections.abc import Sequence

import pypto.language as pl
import pytest
from pypto import DataType, ir


def _assign_stmts(func: ir.Function) -> list[ir.AssignStmt]:
    """Return the function body's top-level AssignStmts, in source order."""
    body = func.body
    stmts = list(body.stmts) if isinstance(body, ir.SeqStmts) else [body]
    assigns = [s for s in stmts if isinstance(s, ir.AssignStmt)]
    assert assigns, "expected at least one binding in the function body"
    return assigns


def _assert_same_exprs(actual: Sequence[ir.Expr], expected: Sequence[ir.Expr]) -> None:
    """Assert two expression lists hold the same nodes, compared by identity.

    ``Expr.__eq__`` is overloaded to build an IR ``Eq`` node rather than return a
    bool, so ``actual == expected`` on lists of ``Expr`` reports equality even
    when the elements differ (a truthy ``Eq`` node). Identity is what these tests
    mean anyway: the parser must reuse the very same ``Var``, not a copy.
    """
    actual, expected = list(actual), list(expected)
    assert len(actual) == len(expected), f"expected {len(expected)} elements, got {len(actual)}"
    for i, (got, want) in enumerate(zip(actual, expected, strict=True)):
        assert got is want, f"element {i}: {got} is not the expected {want}"


def _make_tuple(stmt: ir.AssignStmt) -> ir.MakeTuple:
    """Assert a binding's RHS is a tuple literal and return it."""
    value = stmt.value
    assert isinstance(value, ir.MakeTuple), (
        f"{stmt.var.name_hint} is {type(value).__name__}, expected MakeTuple"
    )
    return value


def _get_item(stmt: ir.AssignStmt) -> ir.TupleGetItemExpr:
    """Assert a binding's RHS is a tuple subscript and return it."""
    value = stmt.value
    assert isinstance(value, ir.TupleGetItemExpr), (
        f"{stmt.var.name_hint} is {type(value).__name__}, expected TupleGetItemExpr"
    )
    return value


def _scalar_dtype(stmt: ir.AssignStmt) -> DataType:
    """Assert a binding's inferred type is a scalar and return its dtype."""
    var_type = stmt.var.type
    assert isinstance(var_type, ir.ScalarType), (
        f"{stmt.var.name_hint} has type {type(var_type).__name__}, expected ScalarType"
    )
    return var_type.dtype


def _tuple_type(stmt: ir.AssignStmt) -> ir.TupleType:
    """Assert a binding's inferred type is a TupleType and return it."""
    var_type = stmt.var.type
    assert isinstance(var_type, ir.TupleType), (
        f"{stmt.var.name_hint} has type {type(var_type).__name__}, expected TupleType"
    )
    return var_type


class TestTupleLiteralParsing:
    """Tests for parsing tuple literals (x, y, z)."""

    def test_parse_empty_tuple(self):
        """Test parsing empty tuple literal."""

        @pl.function
        def func():
            _ = ()

        (binding,) = _assign_stmts(func)
        assert len(_make_tuple(binding).elements) == 0
        assert list(_tuple_type(binding).types) == []

    def test_parse_tuple_with_two_elements(self):
        """Test parsing tuple with two elements."""

        @pl.function
        def func(x: pl.Tensor[[10], pl.FP32], y: pl.Scalar[pl.INT64]):
            _ = (x, y)

        (binding,) = _assign_stmts(func)
        # Elements are the parameters themselves, in order
        _assert_same_exprs(_make_tuple(binding).elements, func.params)
        element_types = list(_tuple_type(binding).types)
        assert isinstance(element_types[0], ir.TensorType)
        assert isinstance(element_types[1], ir.ScalarType)

    def test_parse_tuple_with_constants(self):
        """Test parsing tuple with constant values."""

        @pl.function
        def func():
            _ = (1, 2, 3)

        (binding,) = _assign_stmts(func)
        elements = list(_make_tuple(binding).elements)
        assert len(elements) == 3
        for element, expected in zip(elements, (1, 2, 3), strict=True):
            assert isinstance(element, ir.ConstInt)
            assert element.value == expected

    def test_parse_nested_tuple(self):
        """Test parsing nested tuples."""

        @pl.function
        def func(x: pl.Scalar[pl.INT64]):
            inner = (x, x)
            _ = (inner, x)

        inner_stmt, outer_stmt = _assign_stmts(func)
        _assert_same_exprs(_make_tuple(inner_stmt).elements, [func.params[0], func.params[0]])

        # The outer literal references the inner binding, not a re-spliced copy
        outer_elements = list(_make_tuple(outer_stmt).elements)
        _assert_same_exprs(outer_elements, [inner_stmt.var, func.params[0]])
        # ...and the nesting shows up in the inferred type
        assert isinstance(list(_tuple_type(outer_stmt).types)[0], ir.TupleType)

    def test_parse_singleton_tuple(self):
        """Test parsing single element tuple."""

        @pl.function
        def func(x: pl.Scalar[pl.INT64]):
            _ = (x,)

        (binding,) = _assign_stmts(func)
        # A trailing comma must build a 1-tuple, not unwrap to the bare element
        _assert_same_exprs(_make_tuple(binding).elements, [func.params[0]])
        assert len(list(_tuple_type(binding).types)) == 1


class TestTupleSubscriptParsing:
    """Tests for parsing tuple subscript access tuple[0]."""

    def test_parse_simple_subscript(self):
        """Test parsing simple tuple subscript - need to create tuple first."""

        @pl.function
        def func(x: pl.Scalar[pl.INT64], y: pl.Scalar[pl.FP32]):
            my_tuple = (x, y)
            _first = my_tuple[0]
            _second = my_tuple[1]

        tuple_stmt, first_stmt, second_stmt = _assign_stmts(func)

        first = _get_item(first_stmt)
        assert first.index == 0
        assert first.tuple is tuple_stmt.var

        second = _get_item(second_stmt)
        assert second.index == 1
        assert second.tuple is tuple_stmt.var

        # Each projection carries the corresponding element type
        assert _scalar_dtype(first_stmt) == pl.INT64
        assert _scalar_dtype(second_stmt) == pl.FP32

    def test_parse_nested_subscript(self):
        """Test parsing nested tuple subscript."""

        @pl.function
        def func(x: pl.Scalar[pl.INT64], y: pl.Scalar[pl.FP32]):
            inner = (x, x)
            nested = (inner, y)
            _first = nested[0]
            _inner_second = nested[0][1]

        _, nested_stmt, first_stmt, inner_second_stmt = _assign_stmts(func)

        # nested[0] projects the inner tuple, so its type is still a tuple
        first = _get_item(first_stmt)
        assert first.index == 0
        assert first.tuple is nested_stmt.var
        assert isinstance(first_stmt.var.type, ir.TupleType)

        # nested[0][1] chains a second projection down to a scalar
        inner_second = _get_item(inner_second_stmt)
        assert inner_second.index == 1
        inner_tuple = inner_second.tuple
        assert isinstance(inner_tuple, ir.TupleGetItemExpr)
        assert inner_tuple.index == 0
        assert _scalar_dtype(inner_second_stmt) == pl.INT64


class TestTupleRoundTrip:
    """Tests for creating and accessing tuples."""

    def test_create_and_access_tuple(self):
        """Test creating tuple and accessing elements."""

        @pl.function
        def func(x: pl.Scalar[pl.INT64], y: pl.Scalar[pl.FP32]):
            my_tuple = (x, y)
            _first = my_tuple[0]
            _second = my_tuple[1]

        tuple_stmt, first_stmt, second_stmt = _assign_stmts(func)

        # Round trip: what goes into the literal comes back out of the subscripts
        _assert_same_exprs(_make_tuple(tuple_stmt).elements, func.params)
        assert _get_item(first_stmt).index == 0
        assert _get_item(second_stmt).index == 1
        assert _scalar_dtype(first_stmt) == pl.INT64
        assert _scalar_dtype(second_stmt) == pl.FP32

    def test_tuple_in_operations(self):
        """Test using tuple elements in operations."""

        @pl.function
        def func(x: pl.Scalar[pl.INT64], y: pl.Scalar[pl.INT64]):
            my_tuple = (x, y)
            # Access tuple elements
            first = my_tuple[0]
            second = my_tuple[1]
            # Store them for verification
            _first = first
            _second = second

        stmts = _assign_stmts(func)
        by_name = {s.var.name_hint: s for s in stmts}

        assert _get_item(by_name["first"]).index == 0
        assert _get_item(by_name["second"]).index == 1
        # Re-binding a projected element forwards the same value, not a fresh read
        assert by_name["_first"].value is by_name["first"].var
        assert by_name["_second"].value is by_name["second"].var


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
