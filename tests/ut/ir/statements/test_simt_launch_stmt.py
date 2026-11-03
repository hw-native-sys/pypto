# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import pytest
from pypto import DataType, backend, ir
from pypto.backend import BackendType
from pypto.backend.pto_backend import generate
from pypto.ir.op import simt
from pypto.language.parser import parse_program
from pypto.pypto_core import passes

SPAN = ir.Span.unknown()


def integer(value):
    return ir.ConstInt(value, DataType.INT32, SPAN)


def make_program(storage="tensor", direction=ir.ParamDirection.InOut, dims=(64, 1, 1), same_result=False):
    ty = (
        ir.TensorType([64], DataType.FP32)
        if storage == "tensor"
        else ir.TileType([64], DataType.FP32, memory_space=ir.MemorySpace.Vec)
    )
    src, dst = (ir.Var(name, ty, SPAN) for name in ("src", "dst"))
    idx = ir.Var("idx", ir.ScalarType(DataType.INT32), SPAN)
    value = ir.Var("value", ir.ScalarType(DataType.FP32), SPAN)
    helper = ir.Function(
        "helper",
        [src, (dst, direction)],
        [],
        ir.SeqStmts(
            [
                ir.AssignStmt(idx, simt.thread_idx(ir.SimtAxis.X), SPAN),
                ir.AssignStmt(value, simt.load(src, [idx]), SPAN),
                ir.EvalStmt(simt.store(dst, [idx], value), SPAN),
            ],
            SPAN,
        ),
        SPAN,
        ir.FunctionType.SIMT,
    )
    x, y = (ir.Var(name, ty, SPAN) for name in ("x", "y"))
    out = y if same_result else ir.Var("updated", ty, SPAN)
    launch = ir.SimtLaunchStmt(ir.GlobalVar("helper"), [integer(d) for d in dims], [x, y], [out], [1], SPAN)
    kernel = ir.Function(
        "kernel",
        [x, y],
        [ty],
        ir.SeqStmts([launch, ir.ReturnStmt([out], SPAN)], SPAN),
        SPAN,
        ir.FunctionType.AIV,
    )
    return ir.Program([helper, kernel], "SimtProgram", SPAN), launch


def require_function(program: ir.Program, name: str) -> ir.Function:
    function = program.get_function(name)
    assert function is not None
    return function


def storage_args(launch: ir.SimtLaunchStmt) -> list[ir.Var]:
    args: list[ir.Var] = []
    for arg in launch.args:
        assert isinstance(arg, ir.Var)
        args.append(arg)
    return args


def verify(program, *properties):
    selected = passes.IRPropertySet()
    for prop in properties or (passes.IRProperty.SimtValid,):
        selected.insert(prop)
    return passes.PropertyVerifierRegistry.verify(selected, program)


@pytest.mark.parametrize("storage", ["tensor", "tile"])
def test_program_round_trips_and_defs(storage):
    program, launch = make_program(storage)
    assert not verify(
        program, passes.IRProperty.SimtValid, passes.IRProperty.SSAForm, passes.IRProperty.UseAfterDef
    )
    ir.verify_structural_bindings(program)
    restored = ir.deserialize(ir.serialize(program))
    ir.assert_structural_equal(program, restored)
    assert ir.structural_hash(program) == ir.structural_hash(restored)
    printed = ir.python_print(program)
    assert "FunctionType.SIMT" in printed
    assert "SimtAxis.X" in printed
    assert "simt.launch" in printed
    reparsed = parse_program(printed)
    ir.assert_structural_equal(program, reparsed)
    ir.verify_structural_bindings(reparsed)
    clone, mapping = ir.deep_clone(launch)
    assert isinstance(clone, ir.SimtLaunchStmt)
    assert clone.results[0] is not launch.results[0]
    assert clone.args[1] is launch.args[1]
    assert len(mapping) == 1
    ir.assert_structural_equal(launch, clone, enable_auto_mapping=True)


def test_access_summary_and_physical_store_effect():
    program, _ = make_program()
    helper = require_function(program, "helper")
    access = ir.analyze_simt_access(helper)
    assert [(a.reads, a.writes) for a in access] == [(True, False), (False, True)]
    assert isinstance(helper.body, ir.SeqStmts)
    store = helper.body.stmts[-1]
    assert isinstance(store, ir.EvalStmt)
    assert isinstance(store.expr.type, ir.VoidType)
    assert not ir.is_incore_type(ir.FunctionType.SIMT)
    assert ir.is_device_function_type(ir.FunctionType.SIMT)
    assert ir.get_op_arg_effect("simt.store", 0) == ir.ArgEffect.Write


def test_ssa_launch_definitions_follow_argument_uses():
    program, _ = make_program(same_result=True)
    after = passes.convert_to_ssa()(program)
    body = require_function(after, "kernel").body
    assert isinstance(body, ir.SeqStmts)
    launch = body.stmts[0]
    assert isinstance(launch, ir.SimtLaunchStmt)
    ret = body.stmts[-1]
    assert isinstance(ret, ir.ReturnStmt)
    assert launch.results[0] is not launch.args[1]
    assert ret.value[0] is launch.results[0]
    assert not verify(after, passes.IRProperty.SSAForm, passes.IRProperty.UseAfterDef)


@pytest.mark.parametrize("dims", [(0, 1, 1), (-1, 1, 1), (2**31, 1, 1), (65536, 65536, 1)])
def test_invalid_dimensions(dims):
    with pytest.raises(ValueError, match="positive constant|INT32_MAX"):
        make_program(dims=dims)


def test_dynamic_and_boolean_dimensions_rejected():
    _, launch = make_program()
    for dim in (ir.ConstBool(True, SPAN), ir.Var("n", ir.ScalarType(DataType.INT32), SPAN)):
        with pytest.raises(ValueError, match="positive constant"):
            ir.SimtLaunchStmt(
                launch.callee, [dim, integer(1), integer(1)], launch.args, launch.results, [1], SPAN
            )


@pytest.mark.parametrize("direction", [ir.ParamDirection.In, ir.ParamDirection.Out])
def test_invalid_helper_direction(direction):
    program, _ = make_program(direction=direction)
    messages = " ".join(d.message for d in verify(program))
    assert "InOut" in messages


def test_result_mapping_and_type_must_match():
    program, launch = make_program()
    wrong = ir.SimtLaunchStmt(launch.callee, launch.block_dims, launch.args, [], [], SPAN)
    kernel = ir.Function("kernel", storage_args(launch), [], wrong, SPAN, ir.FunctionType.AIV)
    messages = " ".join(
        d.message for d in verify(ir.Program([require_function(program, "helper"), kernel], "P", SPAN))
    )
    assert "map exactly" in messages
    with pytest.raises(ValueError, match="preserve"):
        ir.SimtLaunchStmt(
            launch.callee,
            launch.block_dims,
            launch.args,
            [ir.Var("bad", ir.TensorType([32], DataType.FP32), SPAN)],
            [1],
            SPAN,
        )


def test_helper_cannot_return_values_or_be_runtime_task():
    with pytest.raises(ValueError, match="void"):
        ir.Function(
            "bad",
            [],
            [ir.ScalarType(DataType.INT32)],
            ir.ReturnStmt([integer(1)], SPAN),
            SPAN,
            ir.FunctionType.SIMT,
        )
    with pytest.raises(ValueError, match="runtime"):
        ir.Function(
            "bad", [], [], ir.SeqStmts([], SPAN), SPAN, ir.FunctionType.SIMT, level=ir.Level.CORE_GROUP
        )


def test_ordinary_calls_and_nested_launches_rejected():
    program, launch = make_program()
    helper = require_function(program, "helper")
    call = ir.Call(launch.callee, launch.args, {}, SPAN)
    kernel = ir.Function(
        "kernel", storage_args(launch), [], ir.EvalStmt(call, SPAN), SPAN, ir.FunctionType.AIV
    )
    messages = " ".join(d.message for d in verify(ir.Program([helper, kernel], "P", SPAN)))
    assert "SimtLaunchStmt" in messages
    nested = ir.Function("nested", storage_args(launch), [], launch, SPAN, ir.FunctionType.SIMT)
    messages = " ".join(d.message for d in verify(ir.Program([helper, nested], "P", SPAN)))
    assert "nested launches" in messages


@pytest.mark.parametrize(
    "threads,regs",
    [(1, 128), (256, 128), (257, 64), (512, 64), (513, 32), (1024, 32), (1025, 16), (2048, 16)],
)
def test_a5_resource_inference(threads, regs):
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend950)
    try:
        _, launch = make_program(dims=(threads, 1, 1))
        resources = ir.infer_simt_resources(launch)
        assert (resources.max_threads, resources.max_regs) == (threads, regs)
        _, too_large = make_program(dims=(2048, 2, 1))
        with pytest.raises(ValueError, match="limit"):
            ir.infer_simt_resources(too_large)
    finally:
        backend.reset_for_testing()


def test_unsupported_target_and_codegen_fail_explicitly():
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)
    try:
        program, launch = make_program()
        with pytest.raises(ValueError, match="unsupported"):
            ir.infer_simt_resources(launch)

        with pytest.raises(NotImplementedError, match="SIMT lowering"):
            generate(program, "/tmp", skip_ptoas=True)
    finally:
        backend.reset_for_testing()


def test_multiple_results_and_alias_rejection():
    program, launch = make_program()
    ty = launch.args[0].type
    a, b = (ir.Var(name, ty, SPAN) for name in ("a", "b"))
    helper = ir.Function(
        "both",
        [(a, ir.ParamDirection.InOut), (b, ir.ParamDirection.InOut)],
        [],
        ir.SeqStmts([], SPAN),
        SPAN,
        ir.FunctionType.SIMT,
    )
    outputs = [ir.Var("x_new", ty, SPAN), ir.Var("y_new", ty, SPAN)]
    both = ir.SimtLaunchStmt(ir.GlobalVar("both"), launch.block_dims, launch.args, outputs, [0, 1], SPAN)
    kernel = ir.Function("kernel", storage_args(launch), [], both, SPAN, ir.FunctionType.AIV)
    p = ir.Program([helper, kernel], "Multiple", SPAN)
    assert not verify(p)
    ir.assert_structural_equal(p, parse_program(ir.python_print(p)))
    with pytest.raises(ValueError, match="increasing"):
        ir.SimtLaunchStmt(both.callee, both.block_dims, both.args, outputs, [1, 0], SPAN)
    aliased = ir.SimtLaunchStmt(
        both.callee, both.block_dims, [both.args[0], both.args[0]], outputs, [0, 1], SPAN
    )
    kernel = ir.Function("kernel", storage_args(launch), [], aliased, SPAN, ir.FunctionType.AIV)
    assert any("alias" in d.message for d in verify(ir.Program([helper, kernel], "Aliased", SPAN)))


def test_simplify_retains_launch_inputs_and_helper_store():
    program, launch = make_program()
    x, y = storage_args(launch)
    src_copy = ir.Var("src_copy", x.type, SPAN)
    copied_launch = ir.SimtLaunchStmt(
        launch.callee, launch.block_dims, [src_copy, y], launch.results, [1], SPAN
    )
    kernel = ir.Function(
        "kernel",
        [x, y],
        [],
        ir.SeqStmts(
            [
                ir.AssignStmt(src_copy, x, SPAN),
                copied_launch,
            ],
            SPAN,
        ),
        SPAN,
        ir.FunctionType.AIV,
    )
    after = passes.simplify()(ir.Program([require_function(program, "helper"), kernel], "Dce", SPAN))
    assert not verify(after, passes.IRProperty.UseAfterDef, passes.IRProperty.SimtValid)
    assert "simt.launch" in ir.python_print(after)
    assert "simt.store" in ir.python_print(after)


def test_simplify_retypes_launch_results_and_downstream_uses():
    before = parse_program("""
@pl.program
class Retype:
    @pl.function(type=pl.FunctionType.SIMT)
    def helper(self, x: pl.InOut[pl.Tensor[[8, 4], pl.FP32]]):
        pl.simt.store(x, (0, 0), pl.const(1.0, pl.FP32))

    @pl.function(type=pl.FunctionType.AIV)
    def kernel(self, x: pl.Tensor[[8, 4], pl.FP32]) -> pl.Tensor[[8, 4], pl.FP32]:
        view = pl.tensor.view(x, layout=pl.TensorLayout.ND)
        first = pl.simt.launch(self.helper, view, block_dims=(32, 1, 1), result_param_indices=(0,))
        second = pl.simt.launch(self.helper, first, block_dims=(32, 1, 1), result_param_indices=(0,))
        return second
""")
    # The identity view has explicit strides; folding it adopts x's bare type.
    # Both launch results and the final return must follow the retyped storage.
    before_body = require_function(before, "kernel").body
    assert isinstance(before_body, ir.SeqStmts)
    before_launch = before_body.stmts[1]
    assert isinstance(before_launch, ir.SimtLaunchStmt)
    before_type = before_launch.results[0].type
    assert isinstance(before_type, ir.TensorType)
    assert before_type.tensor_view is not None

    after = passes.simplify()(before)
    kernel = require_function(after, "kernel")
    body = kernel.body
    assert isinstance(body, ir.SeqStmts)
    launches = [stmt for stmt in body.stmts if isinstance(stmt, ir.SimtLaunchStmt)]
    assert len(launches) == 2
    for launch in launches:
        ir.assert_structural_equal(launch.results[0].type, kernel.params[0].type)
        ir.assert_structural_equal(launch.results[0].type, launch.args[0].type)
    assert launches[1].args[0] is launches[0].results[0]
    ret = body.stmts[-1]
    assert isinstance(ret, ir.ReturnStmt)
    assert ret.value[0] is launches[1].results[0]
    assert not verify(after, passes.IRProperty.SimtValid, passes.IRProperty.UseAfterDef)
    ir.verify_structural_bindings(after)
    ir.assert_structural_equal(after, ir.deserialize(ir.serialize(after)))
    ir.assert_structural_equal(after, parse_program(ir.python_print(after)))


def test_simt_call_attrs_survive_print_parse_round_trip():
    program, _ = make_program()

    class AddMetadata(ir.IRMutator):
        def visit_call(self, call):
            # Cover coordinate queries, value-producing loads and void stores.
            return ir.set_call_attrs(call, {"simt_test_tag": call.op.name, "simt_test_flag": True})

    before = AddMetadata().visit_program(program)
    after = parse_program(ir.python_print(before))
    ir.assert_structural_equal(before, after)
    ir.verify_structural_bindings(after)


def test_launch_outputs_in_branch_and_loop_ssa():
    code = """
@pl.program
class Control:
    @pl.function(type=pl.FunctionType.SIMT)
    def helper(self, x: pl.InOut[pl.Tensor[[64], pl.FP32]]):
        pl.simt.store(x, (0,), pl.const(1.0, pl.FP32))

    @pl.function(type=pl.FunctionType.AIV)
    def kernel(self, x: pl.Tensor[[64], pl.FP32], flag: pl.Scalar[pl.BOOL]) -> pl.Tensor[[64], pl.FP32]:
        for i in pl.range(2):
            if flag:
                x = pl.simt.launch(self.helper, x, block_dims=(64, 1, 1), result_param_indices=(0,))
        return x
"""
    before = parse_program(code)
    after = passes.convert_to_ssa()(before)
    assert not verify(
        after, passes.IRProperty.SSAForm, passes.IRProperty.UseAfterDef, passes.IRProperty.SimtValid
    )
    assert "simt.launch" in ir.python_print(after)
    ir.assert_structural_equal(after, parse_program(ir.python_print(after)))


def test_python_visitor_and_mutator_dispatch():
    _, launch = make_program()

    class Visitor(ir.IRVisitor):
        def __init__(self):
            super().__init__()
            self.launches = []

        def visit_simt_launch_stmt(self, op):
            self.launches.append(op)
            super().visit_simt_launch_stmt(op)

    class Mutator(ir.IRMutator):
        def visit_simt_launch_stmt(self, op):
            return super().visit_simt_launch_stmt(op)

    visitor = Visitor()
    visitor.visit_stmt(launch)
    assert visitor.launches == [launch]
    assert Mutator().visit_stmt(launch) is launch


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
