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
from pypto import DataType, InternalError, backend, ir, passes
from pypto.backend import BackendType
from pypto.backend._ptoas_locate import find_ptoas_binary
from pypto.backend.pto_backend import _generate_kernel_wrapper, _run_ptoas
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
    expected = {
        ir.get_op(f"buffer.{suffix}").name
        for suffix in (
            "add",
            "sub",
            "mul",
            "div",
            "maximum",
            "minimum",
            "neg",
            "abs",
            "relu",
            "sqrt",
            "exp",
            "log",
            "recip",
        )
    }
    assert expected <= recipe_names
    assert len(operations) == len(expected) == 13
    assert Counter(call.op.name for call in operations) == Counter(dict.fromkeys(expected, 1))
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


_SCALAR_NATIVE = ("tadds", "tsubs", "tmuls", "tdivs", "tmaxs", "tmins", "tlrelu", "texpands")


def _scalar_program(scalar_dtype, fill_value):
    operand = "pl.cast(factor, pl.INT64)" if scalar_dtype == "INDEX" else "factor"
    return pl.parse_program(f"""
@pl.program
class ScalarRecipes:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, a: pl.Tensor[[16, 32], pl.FP32], factor: pl.Scalar[pl.{scalar_dtype}],
               output: pl.Out[pl.Tensor[[16, 32], pl.FP32]]) -> pl.Tensor[[16, 32], pl.FP32]:
        lhs = pl.load(a, [0, 0], [16, 32])
        full = pl.tile.full([16, 32], pl.FP32, {fill_value})
        initialized = pl.tile.add(lhs, full)
        scalar = {operand}
        added = pl.tile.adds(initialized, scalar)
        subtracted = pl.tile.subs(added, 2.5)
        multiplied = pl.tile.muls(subtracted, scalar)
        divided = pl.tile.divs(multiplied, 3)
        maximum = pl.tile.maximums(divided, -2.0)
        minimum = pl.tile.minimums(maximum, 4)
        activated = pl.tile.lrelu(minimum, 0.125)
        result = pl.store(activated, [0, 0], output)
        return result
""")


def _lower_scalars(scalar_dtype, fill_value, planner):
    with passes.PassContext([], memory_planner=planner, enable_buffer_ir=True):
        return PassManager.get_strategy(OptimizationStrategy.Default).run_passes(
            _scalar_program(scalar_dtype, fill_value)
        )


@pytest.mark.parametrize("ascend_backend", _BACKENDS, indirect=True)
@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize("scalar_dtype", ["FP32", "INT32", "INT64", "INDEX"])
@pytest.mark.parametrize("fill_value", ["-1.25", "2"])
def test_scalar_recipes_resolve_operands_before_native_emission(
    ascend_backend, planner, scalar_dtype, fill_value
):
    lowered = _lower_scalars(scalar_dtype, fill_value, planner)
    restored = ir.deserialize(ir.serialize(lowered))
    assert isinstance(restored, ir.Program)
    ir.assert_structural_equal(lowered, restored, enable_auto_mapping=True)
    calls = _Calls(restored)
    scalar_names = {
        ir.get_op(f"buffer.{suffix}").name
        for suffix in ("adds", "subs", "muls", "divs", "maximums", "minimums", "lrelu", "full")
    }
    operations = [call for call in calls.calls if call.op.name in scalar_names]
    assert Counter(call.op.name for call in operations) == Counter(dict.fromkeys(scalar_names, 1))
    for call in operations:
        assert isinstance(call.type, ir.VoidType)
        assert isinstance(call.args[-2].type, ir.ScalarType)
        assert call.args[-2].type.dtype == DataType.FP32
        assert isinstance(call.args[-1].type, ir.BufferType)
        assert not call.kwargs  # Logical full shape/dtype are consumed by its destination.
    text = _emit(restored)
    assert text.count(" = pto.alloc_tile ") == len(calls.allocations)
    assert "pto.tmov" not in text
    for mnemonic in _SCALAR_NATIVE:
        assert text.count(f"pto.{mnemonic} ins(") == 1
    full_line = next(line for line in text.splitlines() if "pto.texpands ins(" in line)
    assert re.search(r"ins\([^,]+ : f32\) outs\(", full_line)


@pytest.mark.parametrize("ascend_backend", _BACKENDS, indirect=True)
@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize("scalar_dtype", ["FP32", "INT32", "INT64", "INDEX"])
def test_native_toolchain_accepts_explicit_scalar_recipe_types(
    tmp_path, ascend_backend, planner, scalar_dtype
):
    if find_ptoas_binary() is None:
        pytest.skip("PTOAS is not available")
    source = tmp_path / "scalar_recipes.pto"
    output = tmp_path / "scalar_recipes.cpp"
    source.write_text(_emit(_lower_scalars(scalar_dtype, "2", planner)))
    arch = "a5" if ascend_backend == BackendType.Ascend950 else "a2"
    level = "level2" if planner == passes.MemoryPlanner.PTOAS else "level3"
    _run_ptoas(str(source), str(output), [f"--pto-level={level}", f"--pto-arch={arch}"])
    assert output.is_file()


def _planned_scalar_program():
    captured: list[ir.Program] = []

    def capture(pass_object, program):
        if pass_object.get_name() == "LowerTileToBuffer":
            captured.append(program)

    instrument = passes.CallbackInstrument(before_pass=capture, name="CapturePlannedScalarRecipe")
    with passes.PassContext([instrument], memory_planner=passes.MemoryPlanner.PYPTO, enable_buffer_ir=True):
        PassManager.get_strategy(OptimizationStrategy.Default).run_passes(_scalar_program("INDEX", "2"))
    assert len(captured) == 1
    return captured[0]


class _RewriteRecipe(ir.IRMutator):
    def __init__(self, operation, rewrite):
        super().__init__()
        self.operation = ir.get_op(operation).name
        self.rewrite = rewrite

    def visit_call(self, call):
        return self.rewrite(call) if call.op.name == self.operation else super().visit_call(call)


@pytest.mark.usefixtures("ascend_backend")
@pytest.mark.parametrize(
    "operation,unknown_kwarg", [("tile.adds", False), ("tile.full", False), ("tile.full", True)]
)
def test_lowering_rejects_malformed_raw_recipe_inputs(operation, unknown_kwarg):
    def malformed(call):
        args = list(call.args) if unknown_kwarg else [*call.args, call.args[-1]]
        kwargs = {**call.kwargs, "unexpected": True} if unknown_kwarg else call.kwargs
        return ir.Call(call.op, args, kwargs, None, call.type, call.span)

    malformed_program = _RewriteRecipe(operation, malformed).visit_program(_planned_scalar_program())
    with passes.PassContext([], memory_planner=passes.MemoryPlanner.PYPTO, enable_buffer_ir=True):
        error = ValueError if unknown_kwarg else InternalError
        message = (
            "Unknown kwarg 'unexpected'"
            if unknown_kwarg
            else "malformed Tile elementwise recipe operand count"
        )
        with pytest.raises(error, match=message):
            passes.lower_tile_to_buffer()(malformed_program)


@pytest.mark.usefixtures("ascend_backend")
def test_lowering_makes_raw_index_to_element_conversion_explicit():
    planned = _planned_scalar_program()
    function = next(iter(planned.functions.values()))
    factor = next(param for param in function.params if isinstance(param.type, ir.ScalarType))
    assert isinstance(factor.type, ir.ScalarType) and factor.type.dtype == DataType.INDEX

    def raw_index_operand(call):
        return ir.Call(call.op, [call.args[0], factor], call.kwargs, None, call.type, call.span)

    raw = _RewriteRecipe("tile.adds", raw_index_operand).visit_program(planned)
    with passes.PassContext([], memory_planner=passes.MemoryPlanner.PYPTO, enable_buffer_ir=True):
        lowered = passes.lower_tile_to_buffer()(raw)
    added = next(call for call in _Calls(lowered).calls if call.op.name == ir.get_op("buffer.adds").name)
    scalar = added.args[1]
    assert isinstance(scalar, ir.Cast)
    assert isinstance(scalar.type, ir.ScalarType) and scalar.type.dtype == DataType.FP32
    intermediate = scalar.operand
    assert isinstance(intermediate, ir.Cast)
    assert isinstance(intermediate.type, ir.ScalarType) and intermediate.type.dtype == DataType.INT64
    assert isinstance(intermediate.operand.type, ir.ScalarType)
    assert intermediate.operand.type.dtype == DataType.INDEX
    text = _emit(lowered)
    assert "arith.index_cast" in text and "arith.sitofp" in text


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


@pytest.mark.parametrize("ascend_backend", _BACKENDS, indirect=True)
@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize("suffix", ["get_block_idx", "get_block_num", "get_subblock_idx"])
def test_spmd_queries_use_buffer_value_contract_and_runtime_abi(ascend_backend, planner, suffix, tmp_path):
    program = pl.parse_program(f"""
@pl.program
class SpmdQuery:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, a: pl.Tensor[[128, 32], pl.FP32],
               out: pl.Out[pl.Tensor[[16, 32], pl.FP32]]) -> pl.Tensor[[16, 32], pl.FP32]:
        row = pl.tile.{suffix}() * 16
        value = pl.load(a, [row, 0], [16, 32])
        return pl.store(value, [0, 0], out)
""")
    # The enabled pipeline, including source-level nested queries, must reach
    # explicit Buffer calls after FlattenCallExpr has introduced query SSA.
    with passes.PassContext([], memory_planner=planner, enable_buffer_ir=True):
        lowered = PassManager.get_strategy().run_passes(program)
    restored = ir.deserialize(ir.serialize(lowered))
    assert isinstance(restored, ir.Program)
    ir.assert_structural_equal(lowered, restored, enable_auto_mapping=True)
    queries = [call for call in _Calls(restored).calls if call.op.name == ir.get_op(f"buffer.{suffix}").name]
    assert len(queries) == 1
    assert not queries[0].args and isinstance(queries[0].type, ir.ScalarType)
    assert queries[0].type.dtype == DataType.INDEX
    text = _emit(restored)
    parameter = f"%__pypto_spmd_{suffix.removeprefix('get_')}"
    assert f"{parameter}: i32" in text
    assert f"arith.index_cast {parameter} : i32 to index" in text
    assert f"pto.{suffix}" not in text
    if find_ptoas_binary() is None:
        pytest.skip("PTOAS is not available")
    source, output = tmp_path / "query.pto", tmp_path / "query.cpp"
    source.write_text(text)
    level = "level2" if planner == passes.MemoryPlanner.PTOAS else "level3"
    arch = "a5" if ascend_backend == BackendType.Ascend950 else "a2"
    _run_ptoas(str(source), str(output), [f"--pto-level={level}", f"--pto-arch={arch}"])
    assert output.is_file()
    function = restored.get_function("kernel")
    assert function is not None
    wrapper = _generate_kernel_wrapper(function, output.read_text())
    runtime_name = "get_sub_block_id" if suffix == "get_subblock_idx" else suffix
    assert f"{parameter[1:]} = {runtime_name}(args);" in wrapper
    forwarded = (
        "__pypto_spmd_subblock_idx"
        if suffix == "get_subblock_idx"
        else "__pypto_spmd_block_idx, __pypto_spmd_block_num"
    )
    assert f", {forwarded});" in wrapper


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
