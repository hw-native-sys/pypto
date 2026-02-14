# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for scalar Make helper functions (min_, max_)."""

from typing import cast

import pytest
from pypto import DataType, ir


class TestScalarMakeHelpers:
    """Tests for ir.min_ and ir.max_."""

    def test_min_creation(self):
        """Test ir.min_ creates a Min expression with type promotion."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)

        result = ir.min_(x, y, span)

        assert isinstance(result, ir.Min)
        assert cast(ir.Var, result.left).name == "x"
        assert cast(ir.Var, result.right).name == "y"

    def test_max_creation(self):
        """Test ir.max_ creates a Max expression with type promotion."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)

        result = ir.max_(x, y, span)

        assert isinstance(result, ir.Max)
        assert cast(ir.Var, result.left).name == "x"
        assert cast(ir.Var, result.right).name == "y"

    def test_min_type_promotion(self):
        """Test that min_ promotes operand types (e.g. INT32 + INT64 -> INT64)."""
        span = ir.Span.unknown()
        x = ir.Var("x", ir.ScalarType(DataType.INT32), span)
        y = ir.Var("y", ir.ScalarType(DataType.INT64), span)

        result = ir.min_(x, y, span)

        assert isinstance(result, ir.Min)
        assert result.type == ir.ScalarType(DataType.INT64)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
