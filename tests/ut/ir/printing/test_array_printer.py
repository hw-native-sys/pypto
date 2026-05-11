# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Round-trip tests for ArrayType: print -> reparse -> print should be a fixed point."""

import pypto.language as pl
import pytest
from pypto import DataType, ir
from pypto.ir.printer import python_print


def test_array_type_prints_as_pl_array():
    t = ir.ArrayType(DataType.INT32, 8)
    assert python_print(t) == "pl.Array[8, pl.INT32]"


def test_array_type_prints_uint16():
    t = ir.ArrayType(DataType.UINT16, 32)
    assert python_print(t) == "pl.Array[32, pl.UINT16]"


def test_function_with_array_round_trip():
    """@pl.function with array.create + indexing sugar must print → reparse identically."""
    src = (
        "@pl.function\n"
        "def kernel(x: pl.Tensor[[16], pl.INT32]) -> pl.Tensor[[16], pl.INT32]:\n"
        "    arr: pl.Array[8, pl.INT32] = pl.array.create(8, dtype=pl.INT32)\n"
        "    arr: pl.Array[8, pl.INT32] = pl.array.update_element("
        "arr, 0, pl.const(7, pl.INT32))\n"
        "    v0: pl.Scalar[pl.INT32] = pl.array.get_element(arr, 0)\n"
        "    return x"
    )
    func = pl.parse(src)
    out1 = python_print(func, format=False)
    parsed_again = pl.parse(out1)
    out2 = python_print(parsed_again, format=False)
    assert out1 == out2, f"Round-trip diverged:\n=== first ===\n{out1}\n=== second ===\n{out2}"


def test_indexing_sugar_lowers_to_array_ops():
    """`arr[i] = v` desugars to update_element; `arr[i]` desugars to get_element."""

    @pl.function
    def kernel(x: pl.Tensor[[16], pl.INT32]) -> pl.Tensor[[16], pl.INT32]:
        arr = pl.array.create(4, pl.INT32)
        arr[1] = 7
        _ = arr[1]  # Side-effecting read; surfaces as a get_element call.
        return x

    out = python_print(kernel, format=False)
    assert "pl.array.create(4" in out
    assert "pl.array.update_element(arr, 1, pl.const(7, pl.INT32))" in out
    assert "pl.array.get_element(arr, 1)" in out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
