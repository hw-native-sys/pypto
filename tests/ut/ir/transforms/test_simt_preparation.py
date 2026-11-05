# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""SIMT storage versions must survive preparation and all memory planners."""

import pytest
from pypto import DataType, Error, backend, ir, passes
from pypto.ir.op import simt
from pypto.ir.pass_manager import OptimizationStrategy, PassManager
from pypto.language.parser import parse_program


@pytest.fixture(autouse=True)
def simt_target():
    backend.reset_for_testing()
    backend.set_backend_type(backend.BackendType.Ascend950)
    with passes.PassContext([], enable_buffer_ir=False):
        yield
    backend.reset_for_testing()


def function(program: ir.Program, name: str) -> ir.Function:
    result = program.get_function(name)
    assert result is not None
    return result


def launches(program: ir.Program) -> list[ir.SimtLaunchStmt]:
    result: list[ir.SimtLaunchStmt] = []

    class Collector(ir.IRVisitor):
        def visit_simt_launch_stmt(self, op: ir.SimtLaunchStmt) -> None:
            result.append(op)

    Collector().visit_program(program)
    return result


def test_materialize_tile_output_before_after():
    Before = parse_program("""
@pl.program
class Before:
    @pl.function(type=pl.FunctionType.SIMT)
    def update(self, dst: pl.InOut[pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec]]):
        pl.simt.store(dst, (0, 0), pl.const(9.0, pl.FP32))

    @pl.function(type=pl.FunctionType.AIV)
    def kernel(self, src: pl.Tensor[[8, 8], pl.FP32]):
        old = pl.tile.load(src, [0, 0], [8, 8])
        updated = pl.simt.launch(self.update, old, block_dims=(64, 1, 1), result_param_indices=(0,))
        return old, updated
""")

    Expected = parse_program("""
@pl.program
class Expected:
    @pl.function(type=pl.FunctionType.SIMT)
    def update(self, dst: pl.InOut[pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec]]):
        pl.simt.store(dst, (0, 0), pl.const(9.0, pl.FP32))

    @pl.function(type=pl.FunctionType.AIV)
    def kernel(self, src: pl.Tensor[[8, 8], pl.FP32]):
        old = pl.tile.load(src, [0, 0], [8, 8])
        seed = pl.tile.move(old, target_memory=pl.Mem.Vec, attrs={"simt_output_seed": True})
        updated = pl.simt.launch(self.update, seed, block_dims=(64, 1, 1), result_param_indices=(0,))
        return old, updated
""")

    before = passes.infer_tile_memory_space()(Before)
    expected = passes.infer_tile_memory_space()(Expected)
    after = passes.materialize_simt_outputs()(before)
    ir.assert_structural_equal(after, expected)
    ir.assert_structural_equal(passes.materialize_simt_outputs()(after), after)
    ir.assert_structural_equal(after, ir.deserialize(ir.serialize(after)))
    ir.verify_structural_bindings(after)


@pytest.mark.parametrize(
    "planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.DSA_RP, passes.MemoryPlanner.PTOAS]
)
def test_old_tile_and_launch_result_keep_separate_live_storage(planner):
    Before = parse_program("""
@pl.program
class Before:
    @pl.function(type=pl.FunctionType.SIMT)
    def update(self, dst: pl.InOut[pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec]]):
        pl.simt.store(dst, (0, 0), pl.const(9.0, pl.FP32))

    @pl.function(type=pl.FunctionType.AIV)
    def kernel(self, src: pl.Tensor[[8, 8], pl.FP32]):
        old = pl.tile.load(src, [0, 0], [8, 8])
        updated = pl.simt.launch(self.update, old, block_dims=(64, 1, 1), result_param_indices=(0,))
        temporary = pl.tile.add(old, old)
        return old, updated, temporary
""")

    with passes.PassContext([], memory_planner=planner, enable_buffer_ir=False):
        after = passes.infer_tile_memory_space()(Before)
        after = passes.materialize_simt_outputs()(after)
        after = passes.init_mem_ref()(after)
        after = passes.materialize_semantic_aliases()(after)
        after = passes.memory_reuse()(after)
        after = passes.allocate_memory_addr()(after)

    helper = function(after, "update")
    assert isinstance(helper.params[0].type, ir.TileType)
    assert helper.params[0].type.memref is None
    launch = launches(after)[0]
    body = function(after, "kernel").body
    assert isinstance(body, ir.SeqStmts)
    ret = body.stmts[-1]
    assert isinstance(ret, ir.ReturnStmt)
    old, updated, temporary = ret.value
    assert updated is launch.results[0]
    assert isinstance(old.type, ir.TileType)
    assert isinstance(updated.type, ir.TileType)
    assert isinstance(temporary.type, ir.TileType)
    assert isinstance(launch.args[0].type, ir.TileType)
    old_memory = old.type.memref
    updated_memory = updated.type.memref
    temporary_memory = temporary.type.memref
    assert old_memory is not None and updated_memory is not None and temporary_memory is not None
    assert updated_memory is launch.args[0].type.memref
    assert old_memory.base_ is not updated_memory.base_
    assert temporary_memory.base_ is not updated_memory.base_
    ir.verify_structural_bindings(after)


def test_gm_results_alias_the_input_without_a_copy():
    Before = parse_program("""
@pl.program
class Before:
    @pl.function(type=pl.FunctionType.SIMT)
    def update(self, dst: pl.InOut[pl.Tensor[[8, 8], pl.FP32]]):
        pl.simt.store(dst, (0, 0), pl.const(9.0, pl.FP32))

    @pl.function(type=pl.FunctionType.AIV)
    def kernel(self, dst: pl.Tensor[[8, 8], pl.FP32]):
        updated = pl.simt.launch(self.update, dst, block_dims=(64, 1, 1), result_param_indices=(0,))
        return updated
""")

    materialized = passes.materialize_simt_outputs()(Before)
    ir.assert_structural_equal(materialized, Before)
    after = passes.init_mem_ref()(materialized)
    launch = launches(after)[0]
    argument_type = launch.args[0].type
    result_type = launch.results[0].type
    assert isinstance(argument_type, ir.TensorType)
    assert isinstance(result_type, ir.TensorType)
    assert result_type.memref is argument_type.memref
    helper = function(after, "update")
    assert isinstance(helper.params[0].type, ir.TensorType)
    assert helper.params[0].type.memref is None


def test_specializations_share_identical_dims_and_keep_full_block_shape():
    Before = parse_program("""
@pl.program
class Before:
    @pl.function(type=pl.FunctionType.SIMT)
    def update(self, dst: pl.InOut[pl.Tensor[[8, 8], pl.FP32]]):
        pl.simt.store(dst, (0, 0), pl.const(9.0, pl.FP32))

    @pl.function(type=pl.FunctionType.AIV)
    def kernel(self, a: pl.Tensor[[8, 8], pl.FP32], b: pl.Tensor[[8, 8], pl.FP32]):
        first = pl.simt.launch(self.update, a, block_dims=(64, 1, 1), result_param_indices=(0,))
        second = pl.simt.launch(self.update, b, block_dims=(8, 8, 1), result_param_indices=(0,))
        third = pl.simt.launch(self.update, first, block_dims=(64, 1, 1), result_param_indices=(0,))
        return third, second
""")

    after = passes.specialize_simt_helpers()(Before)
    first, second, third = launches(after)
    assert first.callee.name == third.callee.name
    assert first.callee.name != second.callee.name
    helper = function(after, first.callee.name)
    assert helper.attrs["simt_max_threads"] == 64
    assert helper.attrs["simt_max_regs"] == 128
    assert helper.attrs["simt_block_dims"] == [64, 1, 1]
    assert helper.attrs["simt_target"] == "a5"
    ir.assert_structural_equal(after, passes.specialize_simt_helpers()(after))
    ir.assert_structural_equal(after, ir.deserialize(ir.serialize(after)))
    ir.assert_structural_equal(after, parse_program(ir.python_print(after)))
    ir.verify_structural_bindings(after)


def test_dynamic_strides_are_arguments_and_do_not_duplicate_helpers():
    span = ir.Span.unknown()
    plain = ir.TensorType([8, 8], DataType.FP32)
    dst = ir.Var("dst", plain, span)
    helper = ir.Function(
        "update",
        [(dst, ir.ParamDirection.InOut)],
        [],
        ir.EvalStmt(simt.store(dst, [0, 0], ir.ConstFloat(9, DataType.FP32, span)), span),
        span,
        ir.FunctionType.SIMT,
    )
    callers: list[ir.Function] = []
    strides: list[ir.Var] = []
    for suffix in ("a", "b"):
        stride = ir.Var(f"stride_{suffix}", ir.ScalarType(DataType.INDEX), span)
        strides.append(stride)
        actual_type = ir.TensorType(
            [8, 8], DataType.FP32, tensor_view=ir.TensorView(stride=[stride, 1], layout=ir.TensorLayout.ND)
        )
        tensor = ir.Var("tensor", actual_type, span)
        result = ir.Var("result", actual_type, span)
        launch = ir.SimtLaunchStmt(
            ir.GlobalVar("update"),
            [ir.ConstInt(n, DataType.INT32, span) for n in (64, 1, 1)],
            [tensor],
            [result],
            [0],
            span,
        )
        callers.append(
            ir.Function(f"kernel_{suffix}", [tensor, stride], [], launch, span, ir.FunctionType.AIV)
        )
    before = ir.Program([helper, *callers], "Dynamic", span)

    after = passes.specialize_simt_helpers()(before)
    first, second = launches(after)
    assert first.callee.name == second.callee.name
    assert first.args[-1] is strides[0]
    assert second.args[-1] is strides[1]
    specialized = function(after, first.callee.name)
    assert len(specialized.params) == 2
    storage_type = specialized.params[0].type
    assert isinstance(storage_type, ir.TensorType)
    assert storage_type.tensor_view is not None
    assert storage_type.tensor_view.stride[0] is specialized.params[1]
    ir.assert_structural_equal(after, passes.specialize_simt_helpers()(after))
    ir.verify_structural_bindings(after)


def test_simt_gm_update_propagates_inout_to_outer_call():
    Before = parse_program("""
@pl.program
class Before:
    @pl.function(type=pl.FunctionType.SIMT)
    def update(self, dst: pl.InOut[pl.Tensor[[8, 8], pl.FP32]]):
        pl.simt.store(dst, (0, 0), pl.const(9.0, pl.FP32))

    @pl.function(type=pl.FunctionType.AIV)
    def kernel(self, dst: pl.Tensor[[8, 8], pl.FP32]) -> pl.Tensor[[8, 8], pl.FP32]:
        updated = pl.simt.launch(self.update, dst, block_dims=(64, 1, 1), result_param_indices=(0,))
        return updated

    @pl.function
    def main(self, dst: pl.Tensor[[8, 8], pl.FP32]) -> pl.Tensor[[8, 8], pl.FP32]:
        return self.kernel(dst)
""")

    Expected = parse_program("""
@pl.program
class Expected:
    @pl.function(type=pl.FunctionType.SIMT)
    def update(self, dst: pl.InOut[pl.Tensor[[8, 8], pl.FP32]]):
        pl.simt.store(dst, (0, 0), pl.const(9.0, pl.FP32))

    @pl.function(type=pl.FunctionType.AIV)
    def kernel(self, dst: pl.InOut[pl.Tensor[[8, 8], pl.FP32]]) -> pl.Tensor[[8, 8], pl.FP32]:
        updated = pl.simt.launch(self.update, dst, block_dims=(64, 1, 1), result_param_indices=(0,))
        return updated

    @pl.function
    def main(self, dst: pl.Tensor[[8, 8], pl.FP32]) -> pl.Tensor[[8, 8], pl.FP32]:
        return self.kernel(dst, attrs={"arg_directions": [pl.adir.inout]})
""")

    after = passes.derive_call_directions()(Before)
    ir.assert_structural_equal(after, Expected)
    assert ir.get_op_write_channel("simt.store") == ir.WriteChannel.Simt


@pytest.mark.parametrize("enable_buffer_ir", [False, True])
@pytest.mark.parametrize("shape", [(8, 8), (2, 4, 8)])
def test_default_pipeline_preserves_simt_storage_and_helpers(enable_buffer_ir, shape):
    indices = (0,) * len(shape)

    Before = parse_program(f"""
@pl.program
class Before:
    @pl.function(type=pl.FunctionType.SIMT)
    def update(self, dst: pl.InOut[pl.Tile[{shape}, pl.FP32, pl.Mem.Vec]]):
        pl.simt.store(dst, {indices}, pl.const(9.0, pl.FP32))

    @pl.function(type=pl.FunctionType.AIV)
    def kernel(self, src: pl.Tensor[{shape}, pl.FP32], dst: pl.Out[pl.Tensor[{shape}, pl.FP32]]):
        old = pl.tile.load(src, {indices}, {shape})
        updated = pl.simt.launch(self.update, old, block_dims=(64, 1, 1), result_param_indices=(0,))
        result = pl.tile.store(updated, {indices}, dst)
        return result
""")

    with passes.PassContext([], enable_buffer_ir=enable_buffer_ir):
        after = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(Before)
    launch = launches(after)[0]
    helper = function(after, launch.callee.name)
    assert isinstance(helper.params[0].type, ir.TileType)
    assert len(helper.params[0].type.shape) == 2
    assert function(after, "kernel").ir_stage == ir.FunctionIRStage.Functional
    assert helper.attrs["simt_max_threads"] == 64
    ir.verify_structural_bindings(after)


@pytest.mark.parametrize(
    "planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.DSA_RP, passes.MemoryPlanner.PTOAS]
)
@pytest.mark.parametrize("control", ["for", "if", "while"])
def test_old_tile_survives_control_flow_updates(control, planner):
    header = {"for": "for i in pl.range(3):", "if": "if flag:", "while": "while flag:"}[control]
    body = header + (
        "\n            updated = pl.simt.launch(self.update, updated, "
        "block_dims=(64, 1, 1), result_param_indices=(0,))"
    )
    before = parse_program(f"""
@pl.program
class Before:
    @pl.function(type=pl.FunctionType.SIMT)
    def update(self, dst: pl.InOut[pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec]]):
        pl.simt.store(dst, (0, 0), pl.const(9.0, pl.FP32))

    @pl.function(type=pl.FunctionType.AIV)
    def kernel(self, src: pl.Tensor[[8, 8], pl.FP32], flag: pl.Scalar[pl.BOOL]):
        old = pl.tile.load(src, [0, 0], [8, 8])
        updated = old
        {body}
        return old, updated
""")
    with passes.PassContext([], memory_planner=planner, enable_buffer_ir=False):
        after = passes.convert_to_ssa()(before)
        for transform in (
            passes.infer_tile_memory_space,
            passes.materialize_simt_outputs,
            passes.specialize_simt_helpers,
            passes.init_mem_ref,
            passes.materialize_semantic_aliases,
            passes.memory_reuse,
            passes.allocate_memory_addr,
        ):
            after = transform()(after)
    kernel = function(after, "kernel")
    assert isinstance(kernel.body, ir.SeqStmts)
    ret = kernel.body.stmts[-1]
    assert isinstance(ret, ir.ReturnStmt)
    old, updated = ret.value
    assert isinstance(old.type, ir.TileType)
    assert isinstance(updated.type, ir.TileType)
    assert old.type.memref is not None and updated.type.memref is not None
    assert old.type.memref.base_ is not updated.type.memref.base_
    ir.verify_structural_bindings(after)


def test_known_storage_alias_is_rejected_before_copying():
    Before = parse_program("""
@pl.program
class Before:
    @pl.function(type=pl.FunctionType.SIMT)
    def update(
        self,
        src: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec],
        dst: pl.InOut[pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec]],
    ):
        value = pl.simt.load(src, (0, 0))
        pl.simt.store(dst, (0, 0), value)

    @pl.function(type=pl.FunctionType.AIV)
    def kernel(self, src: pl.Tensor[[8, 8], pl.FP32]):
        old = pl.tile.load(src, [0, 0], [8, 8])
        alias = old
        updated = pl.simt.launch(
            self.update, old, alias, block_dims=(64, 1, 1), result_param_indices=(1,)
        )
        return updated
""")

    before = passes.infer_tile_memory_space()(Before)
    with pytest.raises(ValueError, match="known overlapping storage"):
        passes.materialize_simt_outputs()(before)


@pytest.mark.parametrize("dispatch", ["call", "submit"])
def test_gm_pipeline_preserves_return_identity_and_task_dependencies(dispatch):
    first = "first = self.kernel(dst)" if dispatch == "call" else "first, t0 = pl.submit(self.kernel, dst)"
    second = (
        "second = self.kernel(first)" if dispatch == "call" else "second, t1 = pl.submit(self.kernel, first)"
    )
    before = parse_program(f"""
@pl.program
class Before:
    @pl.function(type=pl.FunctionType.SIMT)
    def update(self, dst: pl.InOut[pl.Tensor[[8, 8], pl.FP32]]):
        pl.simt.store(dst, (0, 0), pl.const(9.0, pl.FP32))

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, dst: pl.Tensor[[8, 8], pl.FP32]) -> pl.Tensor[[8, 8], pl.FP32]:
        updated = pl.simt.launch(self.update, dst, block_dims=(64, 1, 1), result_param_indices=(0,))
        return updated

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, dst: pl.Tensor[[8, 8], pl.FP32]) -> pl.Tensor[[8, 8], pl.FP32]:
        {first}
        {second}
        return second
""")
    after = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(before)
    kernel = function(after, "kernel")
    assert kernel.func_type == ir.FunctionType.AIV
    assert kernel.param_directions == [ir.ParamDirection.InOut]
    assert isinstance(kernel.body, ir.SeqStmts)
    ret = kernel.body.stmts[-1]
    assert isinstance(ret, ir.ReturnStmt)
    assert ret.value[0] is kernel.params[0]
    assert len(launches(after)) == 1
    calls: list[ir.Call | ir.Submit] = []

    class Calls(ir.IRVisitor):
        def visit_call(self, op: ir.Call) -> None:
            if op.op.name == "kernel":
                calls.append(op)
            super().visit_call(op)

        def visit_submit(self, op: ir.Submit) -> None:
            if op.op.name == "kernel":
                calls.append(op)
            super().visit_submit(op)

    Calls().visit_stmt(function(after, "main").body)
    assert len(calls) == 2
    assert calls[0].attrs["arg_directions"] == [ir.ArgDirection.InOut]
    assert calls[1].attrs["arg_directions"] == [ir.ArgDirection.InOut]
    after = passes.auto_derive_task_dependencies(analyze_auto_scopes=True)(after)
    calls.clear()
    Calls().visit_stmt(function(after, "main").body)
    assert len(calls[1].attrs["compiler_manual_dep_edges"]) == 1
    ir.verify_structural_bindings(after)


def test_gm_old_handle_reads_stay_after_launch():
    Before = parse_program("""
@pl.program
class Before:
    @pl.function(type=pl.FunctionType.SIMT)
    def update(self, dst: pl.InOut[pl.Tensor[[8, 8], pl.FP32]]):
        pl.simt.store(dst, (0, 0), pl.const(9.0, pl.FP32))

    @pl.function(type=pl.FunctionType.AIV)
    def kernel(self, dst: pl.Tensor[[8, 8], pl.FP32]):
        updated = pl.simt.launch(self.update, dst, block_dims=(64, 1, 1), result_param_indices=(0,))
        loaded = pl.tile.load(dst, [0, 0], [8, 8])
        return updated, loaded
""")

    body = function(Before, "kernel").body
    assert isinstance(body, ir.SeqStmts)
    graph = passes.stmt_dependency_analysis.build_stmt_dependency_graph(body, Before)
    assert body.stmts[0] in graph.get_predecessors(body.stmts[1])


@pytest.mark.parametrize("flow", ["straight", "chain", "for", "while", "if"])
@pytest.mark.parametrize("annotated", [False, True])
@pytest.mark.parametrize("storage", ["direct", "ND", "DN"])
def test_mixed_kernel_gm_launch_results_resolve_to_cube_parameters(flow, annotated, storage):
    """The AIC consumer must borrow dst, not the AIV-only launch result or src."""
    launch = (
        "updated = pl.simt.launch(self.update, src, updated, "
        "block_dims=(64, 1, 1), result_param_indices=(1,))"
    )
    bodies = {
        "straight": f"        {launch}",
        "chain": f"        {launch}\n        {launch}",
        "for": f"        for i in pl.range(2):\n            {launch}",
        "while": (
            f"        i = pl.const(0, pl.INDEX)\n        while i < 2:\n            {launch}\n"
            "            i = i + 1"
        ),
        "if": f"        if flag == 0:\n            {launch}\n        else:\n            {launch}",
    }
    dst_type = "pl.Tensor[[16, 16], pl.FP16]" if storage == "direct" else "pl.Tensor[[256], pl.FP16]"
    if annotated:
        dst_type = f"pl.InOut[{dst_type}]"
    initial = (
        "dst" if storage == "direct" else f"pl.tensor.view(dst, [16, 16], layout=pl.TensorLayout.{storage})"
    )
    before = parse_program(f"""
@pl.program
class Before:
    @pl.function(type=pl.FunctionType.SIMT)
    def update(self, src: pl.Tensor[[16, 16], pl.FP16], dst: pl.InOut[pl.Tensor[[16, 16], pl.FP16]]):
        value = pl.simt.load(src, (0, 0))
        pl.simt.store(dst, (0, 0), value)

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, src: pl.Tensor[[16, 16], pl.FP16], dst: {dst_type},
               rhs: pl.Tensor[[16, 16], pl.FP16], out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
               flag: pl.Scalar[pl.INT32]):
        updated = {initial}
{bodies[flow]}
        left_mat = pl.tile.load(updated, [0, 0], [16, 16], target_memory=pl.Mem.Mat)
        left = pl.tile.move(left_mat, target_memory=pl.Mem.Left)
        right_mat = pl.tile.load(rhs, [0, 0], [16, 16], target_memory=pl.Mem.Mat)
        right = pl.tile.move(right_mat, target_memory=pl.Mem.Right)
        product = pl.tile.matmul(left, right)
        result = pl.tile.store(product, [0, 0], out)
        return result
""")
    after = passes.convert_to_ssa()(before)
    after = passes.infer_tile_memory_space()(after)
    borrowed_type = launches(after)[0].args[1].type
    after = passes.expand_mixed_kernel()(after)

    aic = function(after, "kernel_aic")
    aiv = function(after, "kernel_aiv")
    loads: list[ir.Call] = []
    views: list[ir.Call] = []

    class Loads(ir.IRVisitor):
        def visit_call(self, op: ir.Call) -> None:
            if op.op.name == ir.get_op("tile.load").name:
                loads.append(op)
            if op.op.name == ir.get_op("tensor.view").name:
                views.append(op)
            super().visit_call(op)

    Loads().visit_stmt(aic.body)
    assert len(loads) == 2
    ir.assert_structural_equal(loads[0].args[0].type, borrowed_type)
    if storage == "direct":
        # A retained loop/phi may still name the handle. Trace its returned
        # identity through the public analysis instead of assuming a flat body.
        probe = ir.Function(
            "probe",
            aic.params,
            [loads[0].args[0].type],
            ir.SeqStmts([aic.body, ir.ReturnStmt([loads[0].args[0]], aic.span)], aic.span),
            aic.span,
            ir.FunctionType.AIC,
        )
        assert ir.returned_param_indices(probe, after) == [1]
        assert not views
    else:
        assert len(views) == 1
        assert views[0].args[0] is aic.params[1]
        ir.assert_structural_equal(views[0].type, borrowed_type)
    assert loads[1].args[0] is aic.params[2]
    assert len(launches(ir.Program([aic], "Cube", aic.span))) == 0
    assert len(launches(ir.Program([aiv], "Vector", aiv.span))) == (2 if flow in ("chain", "if") else 1)
    assert "__FREE_VAR" not in ir.python_print(after)
    ir.verify_structural_bindings(after)
    directed = passes.derive_call_directions()(after)
    assert function(directed, "kernel_aiv").param_directions[1] == ir.ParamDirection.InOut
    assert function(directed, "kernel").param_directions[1] == ir.ParamDirection.InOut


def test_mixed_kernel_preserves_branch_local_gm_views():
    before = parse_program("""
@pl.program
class Before:
    @pl.function(type=pl.FunctionType.SIMT)
    def update(self, dst: pl.InOut[pl.Tensor[[16, 16], pl.FP16]]):
        pl.simt.store(dst, (0, 0), pl.const(1.0, pl.FP16))

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, a: pl.Tensor[[256], pl.FP16], b: pl.Tensor[[256], pl.FP16],
               out: pl.Out[pl.Tensor[[16, 16], pl.FP32]], flag: pl.Scalar[pl.INT32]):
        if flag == 0:
            view = pl.tensor.view(a, [16, 16])
            updated = pl.simt.launch(self.update, view, block_dims=(64, 1, 1), result_param_indices=(0,))
        else:
            view = pl.tensor.view(b, [16, 16])
            updated = pl.simt.launch(self.update, view, block_dims=(64, 1, 1), result_param_indices=(0,))
        tile = pl.tile.load(updated, [0, 0], [16, 16], target_memory=pl.Mem.Mat)
        left = pl.tile.move(tile, target_memory=pl.Mem.Left)
        right = pl.tile.move(tile, target_memory=pl.Mem.Right)
        product = pl.tile.matmul(left, right)
        result = pl.tile.store(product, [0, 0], out)
        return result
""")
    after = passes.expand_mixed_kernel()(passes.infer_tile_memory_space()(passes.convert_to_ssa()(before)))
    aic = function(after, "kernel_aic")
    statements = ir.flatten_to_stmts(aic.body)
    branch = next(stmt for stmt in statements if isinstance(stmt, ir.IfStmt))
    assert branch.else_body is not None
    branch_views: list[tuple[ir.Stmt, ir.Var]] = []
    for body, param in ((branch.then_body, aic.params[0]), (branch.else_body, aic.params[1])):
        definitions = [stmt for stmt in ir.flatten_to_stmts(body) if isinstance(stmt, ir.AssignStmt)]
        assert len(definitions) == 1
        call = definitions[0].value
        assert isinstance(call, ir.Call)
        assert call.op.name == ir.get_op("tensor.view").name
        assert call.args[0] is param
        branch_views.append((body, definitions[0].var))
    load = next(
        stmt.value
        for stmt in statements
        if isinstance(stmt, ir.AssignStmt)
        and isinstance(stmt.value, ir.Call)
        and stmt.value.op.name == ir.get_op("tile.load").name
    )
    slot = next(i for i, value in enumerate(branch.return_vars) if load.args[0] is value)
    for body, view in branch_views:
        yielded = ir.flatten_to_stmts(body)[-1]
        assert isinstance(yielded, ir.YieldStmt)
        assert yielded.value[slot] is view
    assert "__FREE_VAR" not in ir.python_print(after)
    ir.verify_structural_bindings(after)


def test_gm_view_launch_without_cube_work_stays_pure_aiv():
    before = parse_program("""
@pl.program
class Before:
    @pl.function(type=pl.FunctionType.SIMT)
    def update(self, dst: pl.InOut[pl.Tensor[[16, 16], pl.FP16]]):
        pl.simt.store(dst, (0, 0), pl.const(1.0, pl.FP16))

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, dst: pl.Tensor[[256], pl.FP16]):
        view = pl.tensor.view(dst, [16, 16])
        updated = pl.simt.launch(self.update, view, block_dims=(64, 1, 1), result_param_indices=(0,))
""")
    after = passes.expand_mixed_kernel()(passes.infer_tile_memory_space()(passes.convert_to_ssa()(before)))
    assert function(after, "kernel").func_type == ir.FunctionType.AIV
    assert not any(f.func_type == ir.FunctionType.AIC for f in after.functions.values())
    assert len(launches(after)) == 1


def test_launch_inside_active_register_scope_is_rejected():
    Before = parse_program("""
@pl.program
class Before:
    @pl.function(type=pl.FunctionType.SIMT)
    def update(self, dst: pl.InOut[pl.Tensor[[8, 8], pl.FP32]]):
        pl.simt.store(dst, (0, 0), pl.const(9.0, pl.FP32))

    @pl.function(type=pl.FunctionType.AIV)
    def kernel(self, dst: pl.Tensor[[8, 8], pl.FP32]):
        updated = pl.simt.launch(self.update, dst, block_dims=(64, 1, 1), result_param_indices=(0,))
        return updated
""")

    kernel = function(Before, "kernel")
    wrapped = ir.Function(
        kernel.name,
        kernel.params,
        [],
        ir.RegScopeStmt("vmi", kernel.body, kernel.span),
        kernel.span,
        kernel.func_type,
    )
    invalid = ir.Program([function(Before, "update"), wrapped], "Invalid", kernel.span)
    properties = passes.IRPropertySet()
    properties.insert(passes.IRProperty.SimtValid)
    with pytest.raises(Error, match="active VMI register scope"):
        passes.run_verifier(properties)(invalid)


@pytest.mark.parametrize("overlap", [False, True])
def test_known_memref_windows_check_all_writable_parameters(overlap):
    span = ir.Span.unknown()
    plain = ir.TileType([8, 8], DataType.FP32, memory_space=ir.MemorySpace.Vec)
    left, right = (ir.Var(name, plain, span) for name in ("left", "right"))
    helper = ir.Function(
        "update",
        [(left, ir.ParamDirection.InOut), (right, ir.ParamDirection.InOut)],
        [],
        ir.SeqStmts(
            [
                ir.EvalStmt(simt.store(left, [0, 0], ir.ConstFloat(1, DataType.FP32, span)), span),
                ir.EvalStmt(simt.store(right, [0, 0], ir.ConstFloat(2, DataType.FP32, span)), span),
            ],
            span,
        ),
        span,
        ir.FunctionType.SIMT,
    )
    base = ir.MemRef("pool").base_
    actuals = [
        ir.Var(
            name,
            ir.TileType(
                [8, 8],
                DataType.FP32,
                memref=ir.MemRef(base, offset, 256, span),
                memory_space=ir.MemorySpace.Vec,
            ),
            span,
        )
        for name, offset in (("a", 0), ("b", 128 if overlap else 256))
    ]
    launch = ir.SimtLaunchStmt(
        ir.GlobalVar("update"),
        [ir.ConstInt(n, DataType.INT32, span) for n in (64, 1, 1)],
        list(actuals),
        [ir.Var(f"out{i}", arg.type, span) for i, arg in enumerate(actuals)],
        [0, 1],
        span,
    )
    before = ir.Program(
        [helper, ir.Function("kernel", actuals, [], launch, span, ir.FunctionType.AIV)], "Windows", span
    )
    if overlap:
        with pytest.raises(ValueError, match="known overlapping storage"):
            passes.materialize_simt_outputs()(before)
    else:
        after = passes.materialize_simt_outputs()(before)
        materialized = launches(after)[0]
        assert all(arg is not original for arg, original in zip(materialized.args, actuals))
        assert materialized.args[0] is not materialized.args[1]
        assert materialized.result_param_indices == [0, 1]


def test_specialization_ignores_allocations_and_distinguishes_layouts():
    span = ir.Span.unknown()
    plain = ir.TileType([8, 8], DataType.FP32, memory_space=ir.MemorySpace.Vec)
    dst = ir.Var("dst", plain, span)
    helper = ir.Function(
        "update",
        [(dst, ir.ParamDirection.InOut)],
        [],
        ir.EvalStmt(simt.store(dst, [0, 0], ir.ConstFloat(1, DataType.FP32, span)), span),
        span,
        ir.FunctionType.SIMT,
    )
    actuals = [
        ir.Var(
            name,
            ir.TileType(
                [8, 8],
                DataType.FP32,
                memref=ir.MemRef(f"pool{i}", i * 256, 256, span),
                tile_view=ir.TileView(blayout=layout),
                memory_space=ir.MemorySpace.Vec,
            ),
            span,
        )
        for i, (name, layout) in enumerate(
            [
                ("a", ir.TileLayout.row_major),
                ("b", ir.TileLayout.row_major),
                ("c", ir.TileLayout.col_major),
            ]
        )
    ]
    statements: list[ir.Stmt] = [
        ir.SimtLaunchStmt(
            ir.GlobalVar("update"),
            [ir.ConstInt(n, DataType.INT32, span) for n in (64, 1, 1)],
            [arg],
            [ir.Var(f"out{i}", arg.type, span)],
            [0],
            span,
        )
        for i, arg in enumerate(actuals)
    ]
    before = ir.Program(
        [
            helper,
            ir.Function("kernel", actuals, [], ir.SeqStmts(statements, span), span, ir.FunctionType.AIV),
        ],
        "Layouts",
        span,
    )
    after = passes.specialize_simt_helpers()(before)
    first, second, third = launches(after)
    assert first.callee.name == second.callee.name
    assert first.callee.name != third.callee.name
    for launch in launches(after):
        specialized = function(after, launch.callee.name)
        assert isinstance(specialized.params[0].type, ir.TileType)
        assert specialized.params[0].type.memref is None


def test_gm_view_alias_is_rejected():
    Before = parse_program("""
@pl.program
class Before:
    @pl.function(type=pl.FunctionType.SIMT)
    def update(self, src: pl.Tensor[[8, 8], pl.FP32], dst: pl.InOut[pl.Tensor[[8, 8], pl.FP32]]):
        value = pl.simt.load(src, (0, 0))
        pl.simt.store(dst, (0, 0), value)

    @pl.function(type=pl.FunctionType.AIV)
    def kernel(self, src: pl.Tensor[[8, 8], pl.FP32]):
        alias = pl.tensor.slice(src, [8, 8], [0, 0])
        updated = pl.simt.launch(
            self.update, src, alias, block_dims=(64, 1, 1), result_param_indices=(1,)
        )
        return updated
""")

    with pytest.raises(ValueError, match="known overlapping storage"):
        passes.materialize_simt_outputs()(Before)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
