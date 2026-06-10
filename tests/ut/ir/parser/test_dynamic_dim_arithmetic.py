# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Parser unit tests for dynamic dimension arithmetic in shape annotations.

Validates that ``pl.dynamic()`` dims multiplied by compile-time constants
resolve to composite ``ir.Mul`` (etc.) expressions in tensor shapes.
"""

import pypto.language as pl
from pypto.pypto_core import ir


def test_dynamic_dim_mul_in_shape():
    """``NR * 64`` in a shape annotation resolves to ir.Mul."""
    NR = pl.dynamic("NR")
    SIZE = 64
    GATHERED = NR * SIZE

    @pl.program
    class TestProg:
        @pl.function
        def func(self, a: pl.Tensor[[NR, 1], pl.FP32]) -> pl.Tensor[[GATHERED, 1], pl.FP32]:
            out: pl.Tensor[[GATHERED, 1], pl.FP32] = pl.create_tensor([GATHERED, 1], dtype=pl.FP32)
            return out

    func = TestProg.get_function("func")
    assert func is not None
    return_type = func.return_types[0]
    assert isinstance(return_type, ir.TensorType)
    # Shape[0] should be DimExpr wrapping Mul(Var("NR"), ConstInt(64)).
    dim0 = return_type.shape[0]
    assert isinstance(dim0, ir.DimExpr), f"expected ir.DimExpr, got {type(dim0).__name__}"
    assert isinstance(dim0.body, ir.Mul), f"expected ir.Mul body, got {type(dim0.body).__name__}"


def test_dynamic_dim_add_in_shape():
    """``NR + 4`` in a shape annotation resolves to ir.Add."""
    NR = pl.dynamic("NR")
    PADDED = NR + 4

    @pl.program
    class TestProg:
        @pl.function
        def func(self, a: pl.Tensor[[NR, 1], pl.FP32]) -> pl.Tensor[[PADDED, 1], pl.FP32]:
            out: pl.Tensor[[PADDED, 1], pl.FP32] = pl.create_tensor([PADDED, 1], dtype=pl.FP32)
            return out

    func = TestProg.get_function("func")
    assert func is not None
    return_type = func.return_types[0]
    dim0 = return_type.shape[0]
    assert isinstance(dim0, ir.DimExpr), f"expected ir.DimExpr, got {type(dim0).__name__}"
    assert isinstance(dim0.body, ir.Add), f"expected ir.Add body, got {type(dim0.body).__name__}"


def test_dynamic_dim_sub_in_shape():
    """``NR - 1`` in a shape annotation resolves to ir.Sub."""
    NR = pl.dynamic("NR")
    TRIMMED = NR - 1

    @pl.program
    class TestProg:
        @pl.function
        def func(self, a: pl.Tensor[[NR, 1], pl.FP32]) -> pl.Tensor[[TRIMMED, 1], pl.FP32]:
            out: pl.Tensor[[TRIMMED, 1], pl.FP32] = pl.create_tensor([TRIMMED, 1], dtype=pl.FP32)
            return out

    func = TestProg.get_function("func")
    assert func is not None
    return_type = func.return_types[0]
    dim0 = return_type.shape[0]
    assert isinstance(dim0, ir.DimExpr), f"expected ir.DimExpr, got {type(dim0).__name__}"
    assert isinstance(dim0.body, ir.Sub), f"expected ir.Sub body, got {type(dim0.body).__name__}"


def test_dynamic_dim_floordiv_in_shape():
    """``NR // 2`` in a shape annotation resolves to ir.FloorDiv."""
    NR = pl.dynamic("NR")
    HALF = NR // 2

    @pl.program
    class TestProg:
        @pl.function
        def func(self, a: pl.Tensor[[NR, 1], pl.FP32]) -> pl.Tensor[[HALF, 1], pl.FP32]:
            out: pl.Tensor[[HALF, 1], pl.FP32] = pl.create_tensor([HALF, 1], dtype=pl.FP32)
            return out

    func = TestProg.get_function("func")
    assert func is not None
    return_type = func.return_types[0]
    dim0 = return_type.shape[0]
    assert isinstance(dim0, ir.DimExpr), f"expected ir.DimExpr, got {type(dim0).__name__}"
    assert isinstance(dim0.body, ir.FloorDiv), f"expected ir.FloorDiv body, got {type(dim0.body).__name__}"


def test_dynamic_dim_mod_in_shape():
    """``NR % 4`` in a shape annotation resolves to ir.FloorMod."""
    NR = pl.dynamic("NR")
    REM = NR % 4

    @pl.program
    class TestProg:
        @pl.function
        def func(self, a: pl.Tensor[[NR, 1], pl.FP32]) -> pl.Tensor[[REM, 1], pl.FP32]:
            out: pl.Tensor[[REM, 1], pl.FP32] = pl.create_tensor([REM, 1], dtype=pl.FP32)
            return out

    func = TestProg.get_function("func")
    assert func is not None
    return_type = func.return_types[0]
    dim0 = return_type.shape[0]
    assert isinstance(dim0, ir.DimExpr), f"expected ir.DimExpr, got {type(dim0).__name__}"
    assert isinstance(dim0.body, ir.FloorMod), f"expected ir.FloorMod body, got {type(dim0.body).__name__}"


def test_static_mul_still_works():
    """``4 * 64`` (pure Python int arithmetic) still resolves to int."""
    GATHERED = 4 * 64

    @pl.program
    class TestProg:
        @pl.function
        def func(self) -> pl.Tensor[[GATHERED, 1], pl.FP32]:
            out: pl.Tensor[[GATHERED, 1], pl.FP32] = pl.create_tensor([GATHERED, 1], dtype=pl.FP32)
            return out

    func = TestProg.get_function("func")
    assert func is not None
    return_type = func.return_types[0]
    dim0 = return_type.shape[0]
    assert isinstance(dim0, ir.ConstInt) and dim0.value == 256


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
