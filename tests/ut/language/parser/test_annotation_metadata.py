# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Annotations validate value types and retain omitted inferred metadata."""

import pypto.language as pl
import pytest
from pypto import ir, passes
from pypto.language.parser.diagnostics import ParserTypeError


@pytest.mark.parametrize(
    ("annotation", "message"),
    [
        ("pl.Tuple[pl.Tensor[[16, 16], pl.FP32]]", "tuple elements"),
        ("pl.Tuple[pl.Tensor[[8, 16], pl.FP32], pl.Scalar[pl.INT32]]", "shape dimension"),
        ("pl.Tuple[pl.Tensor[[16, 16], pl.FP16], pl.Scalar[pl.INT32]]", "dtype"),
        ("pl.Tuple[pl.Tensor[[16, 16], pl.FP32], pl.Tile[[1], pl.INT32]]", "expression has type"),
    ],
)
def test_tuple_annotation_rejects_conflicting_elements(annotation, message):
    source = f"""@pl.program
class Example:
    @pl.function
    def main(self, x: pl.Tensor[[16, 16], pl.FP32], n: pl.Scalar[pl.INT32]):
        result: {annotation} = (x, n)
        return result
"""
    with pytest.raises(ParserTypeError, match=message):
        pl.parse_program(source)


def test_tuple_annotation_preserves_quant_scale_layout():
    @pl.program
    class Example:
        @pl.function(type=pl.FunctionType.InCore)
        def main(self, src: pl.Tensor[[16, 64], pl.BF16]):
            tile = pl.load(src, [0, 0], [16, 64])
            pair: pl.Tuple[pl.Tile[[16, 64], pl.FP8E4M3FN], pl.Tile[[16, 2], pl.FP8E8M0]] = pl.quant_mx(
                tile, group_axis=1, dtype=pl.FP8E4M3FN
            )
            return pair

    function = next(iter(Example.functions.values()))
    assert isinstance(function.body, ir.SeqStmts)
    binding = function.body.stmts[1]
    assert isinstance(binding, ir.AssignStmt)
    for result_type in (binding.var.type, binding.value.type):
        assert isinstance(result_type, ir.TupleType)
        scale = result_type.types[1]
        assert isinstance(scale, ir.TileType)
        assert scale.tile_view is not None
        assert scale.tile_view.fractal == 32


@pytest.mark.parametrize("loop_kind", ["for", "while"])
@pytest.mark.parametrize("nested", [False, True])
def test_loop_annotation_preserves_omitted_tile_view(loop_kind, nested):
    annotation = "pl.Tile[[16, 16], pl.FP32]"
    if nested:
        annotation = f"pl.Tuple[{annotation}, pl.Tuple[{annotation}]]"
    seed = "(tile, (tile,))" if nested else "tile"
    loop = (
        "for i, (acc,) in pl.range(2, init_values=(seed,)):"
        if loop_kind == "for"
        else "for (acc,) in pl.while_(init_values=(seed,)):"
    )
    condition = "            pl.cond(True)\n" if loop_kind == "while" else ""
    source = f"""@pl.program
class Example:
    @pl.function
    def main(self, src: pl.Tensor[[16, 16], pl.FP32]):
        tile = pl.load(src, [0, 0], [16, 16], valid_shape=[8, 16])
        seed = {seed}
        {loop}
{condition}            out: {annotation} = pl.yield_(acc)
        return out
"""
    program = pl.parse_program(source)
    function = next(iter(program.functions.values()))
    assert isinstance(function.body, ir.SeqStmts)
    loop_stmt = function.body.stmts[2]
    assert isinstance(loop_stmt, (ir.ForStmt, ir.WhileStmt))
    assert loop_stmt.return_vars[0].type == loop_stmt.iter_args[0].type
    ir.assert_structural_equal(program, pl.parse_program(str(program)))


def test_tuple_annotation_does_not_inherit_allocations():
    source = """@pl.program
class Example:
    @pl.function
    def main(self, x: pl.Tensor[[4], pl.FP32, pl.MemRef("base", 0, 16)]):
        result: pl.Tuple[pl.Tensor[[4], pl.FP32]] = (x,)
        return result
"""
    program = pl.parse_program(source)
    function = next(iter(program.functions.values()))
    assert isinstance(function.body, ir.SeqStmts)
    binding = function.body.stmts[0]
    assert isinstance(binding, ir.AssignStmt)
    assert isinstance(binding.var.type, ir.TupleType)
    result_type = binding.var.type.types[0]
    assert isinstance(result_type, ir.TensorType)
    assert result_type.memref is None
    ir.assert_structural_equal(program, pl.parse_program(str(program)))


@pytest.mark.parametrize(
    ("annotation", "value"),
    [
        ("pl.Tuple[pl.Scalar[pl.INT64]]", "(0,)"),
        ("pl.Tuple[pl.Scalar[pl.INT64]]", "(n,)"),
        ("pl.Tuple[pl.Scalar[pl.INT32]]", "(n + 1,)"),
        ("pl.Tuple[pl.Scalar[pl.INT64], pl.Tuple[pl.Scalar[pl.INT32]]]", "(0, (n,))"),
        ("pl.Tuple[pl.Scalar[pl.INDEX]]", "(pl.const(1, pl.INT32),)"),
    ],
)
def test_tuple_annotation_converts_scalar_values(annotation, value):
    source = f"""@pl.program
class Example:
    @pl.function
    def main(self, n: pl.Scalar[pl.INDEX]):
        result: {annotation} = {value}
        return result
"""
    program = pl.parse_program(source)
    properties = passes.IRPropertySet()
    properties.insert(passes.IRProperty.AssignTypeSymmetry)
    assert not passes.PropertyVerifierRegistry.verify(properties, program)
    ir.assert_structural_equal(program, pl.parse_program(str(program)))


def test_tuple_annotation_converts_call_results_without_repeating_call():
    source = """@pl.program
class Example:
    @pl.function
    def make(self, n: pl.Scalar[pl.INDEX]) -> pl.Tuple[pl.Scalar[pl.INDEX], pl.Scalar[pl.INDEX]]:
        return (n, n)
    @pl.function
    def main(self, n: pl.Scalar[pl.INDEX]):
        result: pl.Tuple[pl.Scalar[pl.INT64], pl.Scalar[pl.INT32]] = self.make(n)
        return result
"""
    program = pl.parse_program(source)
    properties = passes.IRPropertySet()
    properties.insert(passes.IRProperty.AssignTypeSymmetry)
    assert not passes.PropertyVerifierRegistry.verify(properties, program)

    class Calls(ir.IRVisitor):
        def __init__(self):
            super().__init__()
            self.calls = []

        def visit_call(self, op):
            if op.op.name == "make":
                self.calls.append(op)
            super().visit_call(op)

    calls = Calls()
    calls.visit_program(program)
    assert len(calls.calls) == 1
    assert isinstance(calls.calls[0].type, ir.TupleType)
    assert all(isinstance(t, ir.ScalarType) and t.dtype == pl.INDEX for t in calls.calls[0].type.types)
    ir.assert_structural_equal(program, pl.parse_program(str(program)))
