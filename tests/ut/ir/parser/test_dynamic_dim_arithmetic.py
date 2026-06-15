# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Parser unit tests for DimExpr wrapping of composite dynamic dimensions.

Each test in this file verifies a property that is **only** true with
the DimExpr IR node — it would fail on main (post-#1765) where composite
dims are stored as bare ``Mul`` / ``Add`` expression nodes.

Properties tested:
1. Parser wraps composite dims in ``ir.DimExpr`` (not bare arithmetic)
2. DimExpr survives print → parse round-trip
3. Bare ``pl.dynamic()`` dims are NOT wrapped (only composites are)
"""

import pypto.language as pl
from pypto.pypto_core import ir


def test_composite_dim_is_wrapped_in_dimexpr():
    """``NR * 64`` in a type annotation produces DimExpr, not bare Mul.

    Before DimExpr (main post-#1765), ``pl.Tensor[[NR * 64, 1], ...]``
    stores ``Mul(Var("NR"), ConstInt(64))`` directly in the shape.
    With DimExpr, the parser wraps it: ``DimExpr(Mul(Var("NR"), ConstInt(64)))``.
    This ``isinstance(dim, ir.DimExpr)`` check is the key differentiator.
    """
    NR = pl.dynamic("NR")
    SIZE = 64
    GATHERED = NR * SIZE

    @pl.program
    class TestProg:
        @pl.function
        def func(self, a: pl.Tensor[[NR, 1], pl.FP32]) -> pl.Tensor[[GATHERED, 1], pl.FP32]:
            return a

    func = TestProg.get_function("func")
    assert func is not None
    dim0 = func.return_types[0].shape[0]
    assert isinstance(dim0, ir.DimExpr), f"expected ir.DimExpr, got {type(dim0).__name__}"
    assert isinstance(dim0.body, ir.Mul)
    assert isinstance(dim0.body.left, ir.Var) and dim0.body.left.name_hint == "NR"
    assert isinstance(dim0.body.right, ir.ConstInt) and dim0.body.right.value == 64


def test_dimexpr_survives_print_parse_roundtrip():
    """A DimExpr-wrapped composite dim survives print → parse round-trip.

    Depends on: Python printer's ``VisitExpr_(DimExprPtr)`` unwrapping (to
    print the inner expression) and structural equality's DimExpr unwrap (to
    match the reparsed DimExpr against the original).  Without both, the
    reparsed function would either lose the DimExpr wrapper or fail
    structural comparison.
    """
    NR = pl.dynamic("NR")
    SIZE = 64
    GATHERED = NR * SIZE

    @pl.program
    class TestProg:
        @pl.function
        def func(self, a: pl.Tensor[[NR, 1], pl.FP32]) -> pl.Tensor[[GATHERED, 1], pl.FP32]:
            return a

    func = TestProg.get_function("func")
    assert func is not None

    printed = ir.python_print(func)
    reparsed_func = ir.parse(printed)
    assert reparsed_func is not None

    dim0 = reparsed_func.return_types[0].shape[0]
    assert isinstance(dim0, ir.DimExpr), (
        f"round-tripped dim should be DimExpr, got {type(dim0).__name__}"
    )
    assert isinstance(dim0.body, ir.Mul)


def test_bare_dynvar_not_wrapped_in_dimexpr():
    """A bare ``pl.dynamic()`` dim (no arithmetic) stays as bare Var.

    DimExpr only wraps composite expressions — a lone ``pl.dynamic("NR")``
    in a shape annotation is still a plain ``Var``, not ``DimExpr(Var)``.
    """
    NR = pl.dynamic("NR")

    @pl.program
    class TestProg:
        @pl.function
        def func(self, a: pl.Tensor[[NR, 1], pl.FP32]) -> pl.Tensor[[NR, 1], pl.FP32]:
            return a

    func = TestProg.get_function("func")
    assert func is not None
    dim0 = func.params[0].type.shape[0]
    assert isinstance(dim0, ir.Var), f"bare Var should not be wrapped, got {type(dim0).__name__}"
    assert dim0.name_hint == "NR"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
