# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Public fixed-descriptor vector recipes reach native code without storage repair."""

import re
from collections import Counter

import pypto.language as pl
import pytest
from pypto import DataType, backend, ir, passes
from pypto.backend import BackendType
from pypto.backend._ptoas_locate import find_ptoas_binary
from pypto.backend.pto_backend import _run_ptoas
from pypto.ir.pass_manager import OptimizationStrategy, PassManager
from pypto.pypto_core import codegen

_SPAN = ir.Span.unknown()
_PLANNERS = [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.DSA_RP, passes.MemoryPlanner.PTOAS]
_BACKENDS = [BackendType.Ascend910B, BackendType.Ascend950]
_NATIVE = [
    "tadd",
    "tsub",
    "tmul",
    "tdiv",
    "tmax",
    "tmin",
    "tneg",
    "tabs",
    "trelu",
    "tsqrt",
    "texp",
    "tlog",
    "trecip",
]


def _public_program(high_precision):
    return pl.parse_program(f"""
@pl.program
class VectorRecipes:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, a: pl.Tensor[[16, 32], pl.FP32], b: pl.Tensor[[16, 32], pl.FP32],
               output: pl.Out[pl.Tensor[[16, 32], pl.FP32]]) -> pl.Tensor[[16, 32], pl.FP32]:
        lhs = pl.load(a, [0, 0], [16, 32])
        rhs = pl.load(b, [0, 0], [16, 32])
        added = pl.tile.add(lhs, rhs)
        subtracted = pl.tile.sub(added, rhs)
        multiplied = pl.tile.mul(subtracted, rhs)
        divided = pl.tile.div(multiplied, rhs, high_precision={high_precision})
        maximum = pl.tile.maximum(divided, lhs)
        minimum = pl.tile.minimum(maximum, rhs)
        negated = pl.tile.neg(minimum)
        magnitude = pl.tile.abs(negated)
        positive = pl.tile.relu(magnitude)
        root = pl.tile.sqrt(positive)
        exponential = pl.tile.exp(root)
        logarithm = pl.tile.log(exponential, high_precision={high_precision})
        reciprocal = pl.tile.recip(logarithm, high_precision={high_precision})
        result = pl.store(reciprocal, [0, 0], output)
        return result
""")


class _Calls(ir.IRVisitor):
    def __init__(self, program):
        super().__init__()
        self.calls: list[ir.Call] = []
        self.allocations: list[ir.AssignStmt] = []
        self.visit_program(program)

    def visit_call(self, call):
        assert ir.get_op_ir_stage(call.op.name) == ir.OpIRStage.Buffer
        self.calls.append(call)
        super().visit_call(call)

    def visit_assign_stmt(self, statement):
        if isinstance(statement.var.type, ir.BufferType):
            self.allocations.append(statement)
        super().visit_assign_stmt(statement)


def _lower(high_precision, planner):
    with passes.PassContext([], memory_planner=planner, enable_buffer_ir=True):
        return PassManager.get_strategy(OptimizationStrategy.Default).run_passes(
            _public_program(high_precision)
        )


def _emit(program):
    return codegen.PTOCodegen().generate(program, emit_tile_addr=False, emit_source_loc=False)


@pytest.mark.parametrize("ascend_backend", _BACKENDS, indirect=True)
@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize("high_precision", [False, True])
def test_public_recipes_preserve_explicit_destinations_through_binary_artifacts(
    ascend_backend, planner, high_precision
):
    lowered = _lower(high_precision, planner)
    restored = ir.deserialize(ir.serialize(lowered))
    assert isinstance(restored, ir.Program)
    ir.assert_structural_equal(lowered, restored, enable_auto_mapping=True)
    properties = passes.IRPropertySet()
    properties.insert(passes.IRProperty.BufferIR)
    assert passes.PropertyVerifierRegistry.verify(properties, restored) == []
    calls = _Calls(restored)
    recipe_names = {
        name.replace("tile.", "buffer.", 1) for name in backend.get_buffer_elementwise_recipe_names()
    }
    operations = [call for call in calls.calls if call.op.name in recipe_names]
    assert len(operations) == len(recipe_names) == 13
    assert Counter(call.op.name for call in operations) == Counter(dict.fromkeys(recipe_names, 1))
    handles = {allocation.var.unique_id for allocation in calls.allocations}
    for call in operations:
        destination = call.args[-1]
        assert isinstance(destination, ir.Var)
        assert destination.unique_id in handles and isinstance(call.type, ir.VoidType)
    reciprocal = next(call for call in operations if call.op.name == ir.get_op("buffer.recip").name)
    reciprocal_source, reciprocal_destination = reciprocal.args
    assert isinstance(reciprocal_source, ir.Var) and isinstance(reciprocal_destination, ir.Var)
    assert reciprocal_source.unique_id != reciprocal_destination.unique_id
    text = _emit(restored)
    assert text.count(" = pto.alloc_tile ") == len(calls.allocations)
    assert "pto.tmov" not in text
    assert text.count("pto.tload ins(") == 2 and text.count("pto.tstore ins(") == 1
    for mnemonic in _NATIVE:
        assert text.count(f"pto.{mnemonic} ins(") == 1
        assert not re.search(rf"= pto\.{mnemonic}\b", text)
    assert (" addr = " in text) == (planner != passes.MemoryPlanner.PTOAS)
    for kind in ("div", "log", "recip"):
        assert (f"#pto<{kind}_precision high_precision>" in text) == high_precision
    with passes.PassContext([], memory_planner=planner, enable_buffer_ir=True):
        ir.assert_structural_equal(
            passes.lower_tile_to_buffer()(restored), restored, enable_auto_mapping=True
        )


@pytest.mark.parametrize("ascend_backend", _BACKENDS, indirect=True)
@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize("high_precision", [False, True])
def test_native_toolchain_accepts_the_actual_lowered_recipe_pipeline(
    tmp_path, ascend_backend, planner, high_precision
):
    if find_ptoas_binary() is None:
        pytest.skip("PTOAS is not available")
    source = tmp_path / "vector_recipes.pto"
    output = tmp_path / "vector_recipes.cpp"
    source.write_text(_emit(_lower(high_precision, planner)))
    arch = "a5" if ascend_backend == BackendType.Ascend950 else "a2"
    level = "level2" if planner == passes.MemoryPlanner.PTOAS else "level3"
    _run_ptoas(str(source), str(output), [f"--pto-level={level}", f"--pto-arch={arch}"])
    assert output.is_file()


def _reciprocal_program(source_address=None, destination_address=None, dynamic=False):
    descriptor = ir.BufferType([16, 32], DataType.FP32, ir.MemorySpace.Vec)
    source = ir.Var("source", descriptor, _SPAN)
    destination = ir.Var("destination", descriptor, _SPAN)
    dynamic_address = ir.Var("address", ir.ScalarType(DataType.INDEX), _SPAN)
    statements = []
    for handle, address in ((source, source_address), (destination, destination_address)):
        args: list[ir.Expr] = [ir.MakeTuple([], _SPAN)]
        if address is not None:
            args.append(
                dynamic_address
                if dynamic and handle is source
                else ir.ConstInt(address, DataType.INDEX, _SPAN)
            )
        call = ir.Call(ir.get_op("buffer.alloc"), args, {}, None, descriptor, _SPAN)
        statements.append(ir.AssignStmt(handle, call, _SPAN))
    mutation = ir.Call(ir.get_op("buffer.recip"), [source, destination], {}, None, ir.VoidType(), _SPAN)
    statements.append(ir.EvalStmt(mutation, _SPAN))
    function = ir.Function(
        "kernel",
        [dynamic_address] if dynamic else [],
        [],
        ir.SeqStmts(statements, _SPAN),
        _SPAN,
        type=ir.FunctionType.InCore,
        ir_stage=ir.FunctionIRStage.Buffer,
    )
    return ir.Program([function], "Reciprocal", _SPAN)


@pytest.mark.usefixtures("ascend_backend")
@pytest.mark.parametrize("address", [0, 32, 2016])
def test_reciprocal_rejects_overlap_between_different_placed_handles(address):
    with pytest.raises(ValueError, match="disjoint placed source and destination ranges"):
        _emit(_reciprocal_program(0, address))


@pytest.mark.usefixtures("ascend_backend")
def test_reciprocal_rejects_unproven_dynamic_address_disjointness():
    with pytest.raises(ValueError, match="provably disjoint constant addresses"):
        _emit(_reciprocal_program(0, 4096, dynamic=True))


@pytest.mark.usefixtures("ascend_backend")
@pytest.mark.parametrize("addressed", [False, True])
def test_reciprocal_allows_independent_explicit_destinations(addressed):
    program = _reciprocal_program(0, 2048) if addressed else _reciprocal_program()
    text = _emit(program)
    assert len(re.findall(r" = pto\.alloc_tile ", text)) == 2
    assert "pto.trecip ins(%source : " in text and ") outs(%destination : " in text
    assert "pto.tmov" not in text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
