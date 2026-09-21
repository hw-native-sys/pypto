# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Constant fill accepts the parser's positional constants and Python literals."""

import pypto.language as pl
import pytest
from pypto import DataType, ir, passes
from pypto.ir.op import tile_ops

_SPAN = ir.Span.unknown()


@pytest.mark.parametrize("parsed", [False, True])
@pytest.mark.parametrize("dtype", [DataType.FP16, DataType.FP32, DataType.INT32])
def test_integer_fill_uses_the_requested_scalar_node_kind(dtype, parsed):
    value = ir.ConstInt(2, DataType.INDEX, _SPAN) if parsed else 2
    call = tile_ops.full([16, 32], dtype, value, _SPAN)
    scalar = call.args[1]
    expected = ir.ConstFloat if dtype.is_float() else ir.ConstInt
    assert isinstance(scalar, expected) and scalar.dtype == dtype and scalar.value == 2
    assert isinstance(call.type, ir.TileType) and call.type.dtype == dtype


@pytest.mark.parametrize("value", ["2", "-1.25"])
def test_positional_and_keyword_fill_literals_have_the_same_ir(value):
    def program(arguments):
        return pl.parse_program(f"""
@pl.program
class Fill:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self) -> pl.Tile[[16, 32], pl.FP32]:
        result = pl.tile.full({arguments})
        return result
""")

    positional = program(f"[16, 32], pl.FP32, {value}")
    keyword = program(f"[16, 32], dtype=pl.FP32, value={value}")
    ir.assert_structural_equal(positional, keyword, enable_auto_mapping=True)


def test_full_preserves_an_explicitly_typed_constant():
    scalar = ir.ConstFloat(0.5, DataType.FP16, _SPAN)
    call = tile_ops.full([16, 32], DataType.FP32, scalar, _SPAN)
    assert isinstance(call.args[1], ir.ConstFloat)
    assert call.args[1].dtype == DataType.FP16 and call.args[1].value == 0.5


@pytest.mark.parametrize("keyword", [False, True])
@pytest.mark.parametrize(
    ("tile_dtype", "fill_dtype", "value"),
    [
        ("FP32", "FP16", "0.5"),
        ("FP32", "BF16", "-0.5"),
        ("FP32", "INT32", "2"),
        ("FP16", "FP32", "0.5"),
        ("INT32", "FP32", "0.5"),
        ("FP16", "FP16", "0.5"),
        ("FP32", "FP32", "0.5"),
        ("INT32", "INT32", "2"),
    ],
)
def test_full_typed_fill_roundtrip(tile_dtype, fill_dtype, value, keyword):
    """Printing preserves fill dtype independently of the destination dtype."""
    fill = f"pl.const({value}, pl.{fill_dtype})"
    arguments = f"dtype=pl.{tile_dtype}, value={fill}" if keyword else f"pl.{tile_dtype}, {fill}"
    before = pl.parse_program(f"""
@pl.program
class Fill:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self) -> pl.Tile[[16, 32], pl.{tile_dtype}]:
        result = pl.tile.full([16, 32], {arguments})
        return result
""")
    printed = before.as_python()
    after = pl.parse_program(printed)
    ir.assert_structural_equal(before, after)
    assert f"value={fill}" in printed

    with passes.PassContext([ir.make_roundtrip_instrument()]):
        passes.convert_to_ssa()(before)


def test_full_rejects_runtime_scalar_values():
    value = ir.Var("value", ir.ScalarType(DataType.FP32), _SPAN)
    with pytest.raises(ValueError, match="requires second argument to be a constant value"):
        tile_ops.full([16, 32], DataType.FP32, value, _SPAN)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
