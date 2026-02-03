# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Tests for tuple literal and subscript syntax in parser."""

import pypto.language as pl
from pypto import ir


class TestTupleLiteralParsing:
    """Tests for parsing tuple literals (x, y, z)."""

    def test_parse_empty_tuple(self):
        """Test parsing empty tuple literal."""

        @pl.function
        def func():
            _ = ()

        # Verify function was created
        assert func is not None
        assert isinstance(func, ir.Function)

    def test_parse_tuple_with_two_elements(self):
        """Test parsing tuple with two elements."""

        @pl.function
        def func(x: pl.Tensor[[10], pl.FP32], y: pl.Scalar[pl.INT64]):
            _ = (x, y)

        assert func is not None
        assert isinstance(func, ir.Function)

    def test_parse_tuple_with_constants(self):
        """Test parsing tuple with constant values."""

        @pl.function
        def func():
            _ = (1, 2, 3)

        assert func is not None
        assert isinstance(func, ir.Function)

    def test_parse_nested_tuple(self):
        """Test parsing nested tuples."""

        @pl.function
        def func(x: pl.Scalar[pl.INT64]):
            inner = (x, x)
            _ = (inner, x)

        assert func is not None
        assert isinstance(func, ir.Function)

    def test_parse_singleton_tuple(self):
        """Test parsing single element tuple."""

        @pl.function
        def func(x: pl.Scalar[pl.INT64]):
            _ = (x,)

        assert func is not None
        assert isinstance(func, ir.Function)


class TestTupleSubscriptParsing:
    """Tests for parsing tuple subscript access tuple[0]."""

    def test_parse_simple_subscript(self):
        """Test parsing simple tuple subscript - need to create tuple first."""

        @pl.function
        def func(x: pl.Scalar[pl.INT64], y: pl.Scalar[pl.FP32]):
            my_tuple = (x, y)
            _ = my_tuple[0]
            _ = my_tuple[1]

        assert func is not None
        assert isinstance(func, ir.Function)

    def test_parse_nested_subscript(self):
        """Test parsing nested tuple subscript."""

        @pl.function
        def func(x: pl.Scalar[pl.INT64], y: pl.Scalar[pl.FP32]):
            inner = (x, x)
            nested = (inner, y)
            _ = nested[0]
            _ = nested[0][1]

        assert func is not None
        assert isinstance(func, ir.Function)


class TestTupleRoundTrip:
    """Tests for creating and accessing tuples."""

    def test_create_and_access_tuple(self):
        """Test creating tuple and accessing elements."""

        @pl.function
        def func(x: pl.Scalar[pl.INT64], y: pl.Scalar[pl.FP32]):
            my_tuple = (x, y)
            _ = my_tuple[0]
            _ = my_tuple[1]

        assert func is not None
        assert isinstance(func, ir.Function)

    def test_tuple_in_operations(self):
        """Test using tuple elements in operations."""

        @pl.function
        def func(x: pl.Scalar[pl.INT64], y: pl.Scalar[pl.INT64]):
            my_tuple = (x, y)
            # Access tuple elements
            first = my_tuple[0]
            second = my_tuple[1]
            # Store them for verification
            _ = first
            _ = second

        assert func is not None
        assert isinstance(func, ir.Function)
