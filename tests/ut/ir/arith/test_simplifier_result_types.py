# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Arithmetic rewrites retain explicitly declared scalar result types."""

import pytest
from pypto import DataType, ir
from pypto.arith import CanonicalSimplifier, RewriteSimplifier


@pytest.mark.parametrize("simplifier_type", [RewriteSimplifier, CanonicalSimplifier])
@pytest.mark.parametrize("dtype", [DataType.INT32, DataType.INT64])
@pytest.mark.parametrize("operation", ["identity", "cancel", "constant", "nested"])
def test_rewrite_preserves_result_dtype(simplifier_type, dtype, operation):
    span = ir.Span.unknown()
    index = DataType.INDEX
    x = ir.Var("x", ir.ScalarType(index), span)
    one = ir.ConstInt(1, index, span)
    zero = ir.ConstInt(0, index, span)
    if operation == "identity":
        expr = ir.Add(x, zero, dtype, span)
    elif operation == "cancel":
        expr = ir.Sub(ir.Add(x, one, index, span), one, dtype, span)
    elif operation == "constant":
        expr = ir.Sub(one, one, dtype, span)
    else:
        inner = ir.Sub(ir.Add(x, one, index, span), one, dtype, span)
        expr = ir.Add(inner, ir.ConstInt(0, dtype, span), dtype, span)
    result = simplifier_type()(expr)
    assert result.type.dtype == dtype
    if operation == "constant":
        assert isinstance(result, ir.ConstInt)
        assert result.value == 0
    else:
        ir.assert_structural_equal(result, ir.Cast(x, dtype, span))
    ir.assert_structural_equal(simplifier_type()(result), result)
