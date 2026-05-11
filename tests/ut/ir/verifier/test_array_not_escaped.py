# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Tests for the ArrayNotEscaped property verifier.

ArrayType lives on the on-core scalar register file / C stack. It must not
escape a function — neither as a parameter nor as a return value.
"""

import pypto.language as pl
import pytest
from pypto.pypto_core import passes


def _verify(prog):
    props = passes.IRPropertySet()
    props.insert(passes.IRProperty.ArrayNotEscaped)
    return passes.PropertyVerifierRegistry.verify(props, prog)


def test_array_as_param_is_rejected():
    src = (
        "@pl.program\n"
        "class P:\n"
        "    @pl.function\n"
        "    def bad(self, arr: pl.Array[8, pl.INT32], x: pl.Tensor[[16], pl.INT32]) "
        "-> pl.Tensor[[16], pl.INT32]:\n"
        "        return x\n"
    )
    prog = pl.parse_program(src)
    diags = _verify(prog)
    assert len(diags) == 1
    assert diags[0].rule_name == "ArrayNotEscaped"
    assert "parameter" in diags[0].message
    assert "ArrayType" in diags[0].message


def test_array_as_return_is_rejected():
    src = (
        "@pl.program\n"
        "class P:\n"
        "    @pl.function\n"
        "    def bad(self, x: pl.Tensor[[16], pl.INT32]) -> pl.Array[4, pl.INT32]:\n"
        "        arr = pl.array.create(4, pl.INT32)\n"
        "        return arr\n"
    )
    prog = pl.parse_program(src)
    diags = _verify(prog)
    assert len(diags) == 1
    assert diags[0].rule_name == "ArrayNotEscaped"
    assert "return type" in diags[0].message


def test_local_array_use_is_legal():
    src = (
        "@pl.program\n"
        "class P:\n"
        "    @pl.function\n"
        "    def good(self, x: pl.Tensor[[16], pl.INT32]) -> pl.Tensor[[16], pl.INT32]:\n"
        "        arr = pl.array.create(8, pl.INT32)\n"
        "        arr[0] = 7\n"
        "        v = arr[0]\n"
        "        return x\n"
    )
    prog = pl.parse_program(src)
    diags = _verify(prog)
    assert diags == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
