# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for Var class."""

import pytest
from pypto import DataType, ir


class TestVar:
    """Tests for Var class."""

    def test_var_creation(self):
        """Test creating a Var expression."""
        span = ir.Span("test.py", 1, 1, 1, 5)
        var = ir.Var("x", ir.ScalarType(DataType.INT64), span)

        assert var.name == "x"
        assert var.span.filename == "test.py"

    def test_var_is_expr(self):
        """Test that Var is an instance of Expr."""
        span = ir.Span("test.py", 1, 1, 1, 5)
        var = ir.Var("x", ir.ScalarType(DataType.INT64), span)

        assert isinstance(var, ir.Expr)
        assert isinstance(var, ir.IRNode)

    def test_var_immutability(self):
        """Test that Var attributes are immutable."""
        span = ir.Span("test.py", 1, 1, 1, 5)
        var = ir.Var("x", ir.ScalarType(DataType.INT64), span)

        # Attempting to modify should raise AttributeError
        with pytest.raises(AttributeError):
            var.name = "y"  # type: ignore


if __name__ == "__main__":
    pytest.main(["-v", __file__])
