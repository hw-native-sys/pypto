# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for comparison operation expressions: Eq, Ne, Lt, Le, Gt, Ge."""

from typing import cast

import pytest
from pypto import DataType, ir


class TestComparisonOps:
    """Tests for comparison operations."""

    def test_comparison_ops(self):
        """Test creating comparison expressions."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)

        eq_expr = ir.Eq(x, y, dtype, span)
        assert cast(ir.Var, eq_expr.left).name == "x"

        ne_expr = ir.Ne(x, y, dtype, span)
        assert cast(ir.Var, ne_expr.left).name == "x"

        lt_expr = ir.Lt(x, y, dtype, span)
        assert cast(ir.Var, lt_expr.left).name == "x"

        le_expr = ir.Le(x, y, dtype, span)
        assert cast(ir.Var, le_expr.left).name == "x"

        gt_expr = ir.Gt(x, y, dtype, span)
        assert cast(ir.Var, gt_expr.left).name == "x"

        ge_expr = ir.Ge(x, y, dtype, span)
        assert cast(ir.Var, ge_expr.left).name == "x"


if __name__ == "__main__":
    pytest.main(["-v", __file__])
