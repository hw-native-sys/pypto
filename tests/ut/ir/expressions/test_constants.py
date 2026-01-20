# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for constant expressions: ConstInt, ConstFloat, ConstBool."""

import pytest
from pypto import DataType, ir


class TestConstInt:
    """Tests for ConstInt class."""

    def test_const_creation(self):
        """Test creating a ConstInt expression."""
        span = ir.Span("test.py", 1, 1, 1, 5)
        const = ir.ConstInt(42, DataType.INT64, span)

        assert const.value == 42
        assert const.span.filename == "test.py"

    def test_const_integer(self):
        """Test ConstInt with integer values."""
        span = ir.Span("test.py", 1, 1, 1, 5)
        const = ir.ConstInt(5, DataType.INT64, span)

        assert const.value == 5

    def test_const_is_expr(self):
        """Test that ConstInt is an instance of Expr."""
        span = ir.Span("test.py", 1, 1, 1, 5)
        const = ir.ConstInt(10, DataType.INT64, span)

        assert isinstance(const, ir.Expr)
        assert isinstance(const, ir.IRNode)

    def test_const_immutability(self):
        """Test that ConstInt attributes are immutable."""
        span = ir.Span("test.py", 1, 1, 1, 5)
        const = ir.ConstInt(42, DataType.INT64, span)

        # Attempting to modify should raise AttributeError
        with pytest.raises(AttributeError):
            const.value = 100  # type: ignore


class TestConstBool:
    """Tests for ConstBool class."""

    def test_const_bool_creation_true(self):
        """Test creating a ConstBool expression with True value."""
        span = ir.Span("test.py", 1, 1, 1, 5)
        const = ir.ConstBool(True, span)

        assert const.value is True
        assert const.span.filename == "test.py"
        assert const.dtype == DataType.BOOL

    def test_const_bool_creation_false(self):
        """Test creating a ConstBool expression with False value."""
        span = ir.Span("test.py", 1, 1, 1, 5)
        const = ir.ConstBool(False, span)

        assert const.value is False
        assert const.dtype == DataType.BOOL

    def test_const_bool_is_expr(self):
        """Test that ConstBool is an instance of Expr."""
        span = ir.Span("test.py", 1, 1, 1, 5)
        const = ir.ConstBool(True, span)

        assert isinstance(const, ir.Expr)
        assert isinstance(const, ir.IRNode)

    def test_const_bool_immutability(self):
        """Test that ConstBool attributes are immutable."""
        span = ir.Span("test.py", 1, 1, 1, 5)
        const = ir.ConstBool(True, span)

        # Attempting to modify should raise AttributeError
        with pytest.raises(AttributeError):
            const.value = False  # type: ignore


if __name__ == "__main__":
    pytest.main(["-v", __file__])
