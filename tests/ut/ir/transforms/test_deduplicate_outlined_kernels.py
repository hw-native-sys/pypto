# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Definition reuse preserves dispatches, bindings, and exclusive Group ownership."""

import pypto.language as pl
import pytest
from pypto import DataType as D
from pypto import backend, codegen, ir, passes
from pypto.backend import BackendType
from pypto.ir.op import prefetch_ops, system_ops, tensor
from pypto.ir.pass_manager import OptimizationStrategy, PassManager

S = ir.Span.unknown()


@pytest.mark.parametrize("escape", [None, "return", "argument", "metadata", "signature", "nested_type"])
def test_legacy_statement_results_are_admitted_only_when_discarded(escape):
    typ = ir.TensorType([256], D.INT64)
    functions = []
    for name in ["a", "b"]:
        x = var("workspace", typ)
        call = system_ops.set_ffts(x)
        stmts: list[ir.Stmt] = [ir.EvalStmt(call, S)]
        attrs = None
        returns: list[ir.Type] = []
        # Reuse the exact Call object in both positions to exercise DAG edges,
        # rather than relying on a second node being visited independently.
        if escape == "return":
            stmts.append(ir.ReturnStmt([call], S))
        elif escape == "argument":
            stmts.append(ir.EvalStmt(ir.Call(ir.get_op("system.set_ffts"), [call], call.type, S), S))
        elif escape == "metadata":
            attrs = {"retained_statement": call}
        elif escape == "signature":
            returns = [ir.UnknownType()]
        elif escape == "nested_type":
            returns = [ir.TupleType([ir.UnknownType()])]
        functions.append(
            ir.Function(
                name, [x], returns, ir.SeqStmts(stmts, S), S, type=ir.FunctionType.InCore, attrs=attrs
            )
        )
    source = program([*functions, caller(["a", "b"], typ)])
    result = run(source)
    assert names(result) == ({"a", "main"} if escape is None else {"a", "b", "main"})
    assert len(calls(result)) == 2


def test_unknown_value_result_is_not_assumed_to_be_a_statement():
    typ = ir.TensorType([16], D.FP32)
    functions = []
    for name in ["a", "b"]:
        x = var("x", typ)
        # A registered op with an unresolved value type is still not a known
        # statement-only operation, even when its result is discarded.
        call = ir.Call(ir.get_op("tensor.add"), [x, x], ir.UnknownType(), S)
        functions.append(ir.Function(name, [x], [], ir.EvalStmt(call, S), S, type=ir.FunctionType.InCore))
    source = program([*functions, caller(["a", "b"], typ)])
    assert names(run(source)) == {"a", "b", "main"}


@pytest.mark.parametrize(
    "mode",
    ["discard", "wait", "return_context", "return_event", "metadata", "alias", "parameter", "cross_context"],
)
def test_prefetch_handles_must_stay_local_to_each_kernel_invocation(mode):
    typ = ir.TensorType([64], D.FP32)
    functions = []
    for name in ["a", "b"]:
        x = var("x", typ)
        make_context = prefetch_ops.make_context()
        ctx = var("ctx", make_context.type)
        stmts: list[ir.Stmt] = []
        params = [x]
        attrs = None
        returns: list[ir.Type] = []
        if mode == "parameter":
            params.append(ctx)
        else:
            stmts.append(ir.AssignStmt(ctx, make_context, S))
        event_call = prefetch_ops.async_prefetch(x, ctx)
        event = var("event", event_call.type)
        if mode == "discard":
            stmts.append(ir.EvalStmt(event_call, S))
        else:
            stmts.append(ir.AssignStmt(event, event_call, S))
            session_context = ctx
            if mode == "cross_context":
                session_context = var("other_ctx", make_context.type)
                stmts.append(ir.AssignStmt(session_context, prefetch_ops.make_context(), S))
            session_call = prefetch_ops.session(session_context)
            session = var("session", session_call.type)
            stmts.append(ir.AssignStmt(session, session_call, S))
            stmts.append(ir.EvalStmt(prefetch_ops.wait(event, session), S))
        if mode in ("return_context", "return_event"):
            escaped = ctx if mode == "return_context" else event
            returns = [escaped.type]
            stmts.append(ir.ReturnStmt([escaped], S))
        elif mode == "metadata":
            attrs = {"escaped_context": ctx}
        elif mode == "alias":
            stmts.append(ir.AssignStmt(var("alias", ctx.type), ctx, S))
        functions.append(
            ir.Function(
                name, params, returns, ir.SeqStmts(stmts, S), S, type=ir.FunctionType.InCore, attrs=attrs
            )
        )
    source = program([*functions, caller(["a", "b"], typ)])
    result = run(source)
    assert names(result) == ({"a", "main"} if mode in ("discard", "wait") else {"a", "b", "main"})
    assert len(calls(result)) == 2


def var(name, typ=None):
    return ir.Var(name, typ or ir.ScalarType(D.INT64), S)


def kernel(name, *, typ=None, attrs=None, capture=None, kind=ir.FunctionType.InCore):
    x = var(name + "_arg", typ)
    value = x if capture is None else capture
    return ir.Function(name, [x], [value.type], ir.ReturnStmt([value], S), S, type=kind, attrs=attrs)


def caller(names, typ=None):
    x = var("actual", typ)
    parameter = x
    statements: list[ir.Stmt] = []
    for i, name in enumerate(names):
        y = var(f"result_{i}", x.type)
        statements.append(ir.AssignStmt(y, ir.Call(ir.GlobalVar(name), [x], x.type, S), S))
        x = y
    statements.append(ir.ReturnStmt([x], S))
    return ir.Function(
        "main",
        [parameter],
        [x.type],
        ir.SeqStmts(statements, S),
        S,
        type=ir.FunctionType.Orchestration,
    )


def program(functions):
    return ir.Program(functions, "test", S)


def run(prog):
    # Hand-built fixtures deliberately exercise unsupported/malformed inputs.
    with passes.PassContext([]):
        return passes.deduplicate_outlined_kernels()(prog)


def names(prog):
    return {f.name for f in prog.functions.values()}


def calls(prog, function="main"):
    result = []

    class Collector(ir.IRVisitor):
        def visit_call(self, op):
            if isinstance(op.op, ir.GlobalVar):
                result.append(op)
            super().visit_call(op)

        def visit_submit(self, op):
            result.append(op)
            super().visit_submit(op)

    func = prog.get_function(function)
    assert func is not None
    Collector().visit_function(func)
    return result


def test_alpha_equivalent_definitions_preserve_two_executions_and_input():
    source = program([kernel("b"), caller(["b", "a"]), kernel("a")])
    before = ir.structural_hash(source)
    result = run(source)
    assert names(result) == {"a", "main"}
    assert result.get_global_var("a") is source.get_global_var("a")
    assert result.get_global_var("main") is source.get_global_var("main")
    assert [c.op.name for c in calls(result)] == ["a", "a"]
    main = result.get_function("main")
    assert main is not None and isinstance(main.body, ir.SeqStmts)
    assignment = main.body.stmts[0]
    assert isinstance(assignment, ir.AssignStmt)
    assert calls(result)[1].args[0] is assignment.var
    assert ir.structural_hash(source) == before
    ir.assert_structural_equal(run(result), result)
    reversed_source = program(list(reversed(list(source.functions.values()))))
    ir.assert_structural_equal(run(reversed_source), result)


@pytest.mark.parametrize("attrs", [{"cache_policy": "different"}, {"split": 1}])
def test_semantic_attributes_prevent_merging(attrs):
    result = run(program([kernel("a"), kernel("b", attrs=attrs), caller(["a", "b"])]))
    assert names(result) == {"a", "b", "main"}


def test_same_free_capture_is_not_a_closed_kernel():
    capture = var("free")
    result = run(program([kernel("a", capture=capture), kernel("b", capture=capture), caller(["a", "b"])]))
    assert names(result) == {"a", "b", "main"}


def test_capture_hidden_in_attribute_is_rejected():
    capture = var("hidden")
    result = run(
        program(
            [
                kernel("a", attrs={"capture": capture}),
                kernel("b", attrs={"capture": capture}),
                caller(["a", "b"]),
            ]
        )
    )
    assert names(result) == {"a", "b", "main"}


def test_unreferenced_entries_are_preserved():
    result = run(program([kernel("a"), kernel("b"), kernel("exposed"), caller(["a", "b"])]))
    assert names(result) == {"a", "exposed", "main"}


@pytest.mark.parametrize("attr", ["external_source", "builtin_template_dir"])
def test_external_and_builtin_are_not_merged(attr):
    result = run(
        program(
            [
                kernel("a", attrs={attr: "kernel.cpp"}),
                kernel("b", attrs={attr: "kernel.cpp"}),
                caller(["a", "b"]),
            ]
        )
    )
    assert names(result) == {"a", "b", "main"}


def test_alpha_equivalent_shapes_merge_without_losing_repeated_symbol_constraints():
    n, m, k = (
        var("n", ir.ScalarType(D.INDEX)),
        var("m", ir.ScalarType(D.INDEX)),
        var("k", ir.ScalarType(D.INDEX)),
    )
    a = kernel("a", typ=ir.TensorType([n, n], D.FP32))
    b = kernel("b", typ=ir.TensorType([m, m], D.FP32))
    c = kernel("c", typ=ir.TensorType([m, k], D.FP32))
    source = program([a, b, c, caller(["a", "b", "c"], ir.TensorType([16, 16], D.FP32))])
    result = run(source)
    assert ir.structural_equal(a, b, enable_auto_mapping=False)
    assert names(result) == {"a", "c", "main"}
    assert calls(result)[1].type is calls(source)[1].type


@pytest.mark.parametrize("shared_symbol", [False, True])
@pytest.mark.parametrize("use_submit", [False, True])
@pytest.mark.parametrize("grouped", [False, True])
def test_dynamic_output_allocations_preserve_each_calls_symbols(shared_symbol, use_submit, grouped):
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910)
    n = var("N", ir.ScalarType(D.INDEX))
    m = n if shared_symbol else var("M", ir.ScalarType(D.INDEX))
    functions: list[ir.Function] = []
    inputs: list[ir.Var] = []
    outputs: list[ir.Expr] = []
    statements: list[ir.Stmt] = []
    for name, dim in [("first", n), ("second", m)]:
        typ = ir.TensorType([dim], D.FP32)
        x, y = var("x", typ), var("y", typ)
        body = ir.SeqStmts([ir.AssignStmt(y, tensor.add(x, x), S), ir.ReturnStmt([y], S)], S)
        functions.append(ir.Function(name, [x], [typ], body, S, type=ir.FunctionType.InCore))
        if grouped:
            arg, result = var("arg", typ), var("result", typ)
            forward = ir.Call(ir.GlobalVar(name), [arg], typ, S)
            name = "group_" + name
            functions.append(
                ir.Function(
                    name,
                    [arg],
                    [typ],
                    ir.SeqStmts([ir.AssignStmt(result, forward, S), ir.ReturnStmt([result], S)], S),
                    S,
                    type=ir.FunctionType.Group,
                )
            )
        actual = var("input_" + name, typ)
        if use_submit:
            result_type = ir.TupleType([typ, ir.ScalarType(D.TASK_ID)])
            dispatch = ir.Submit(ir.GlobalVar(name), [actual], [], {}, {}, result_type, S)
        else:
            result_type = typ
            dispatch = ir.Call(ir.GlobalVar(name), [actual], result_type, S)
        out = var("result_" + name, result_type)
        inputs.append(actual)
        outputs.append(out)
        statements.append(ir.AssignStmt(out, dispatch, S))
    statements.append(ir.ReturnStmt(outputs, S))
    functions.append(
        ir.Function(
            "main",
            inputs,
            [out.type for out in outputs],
            ir.SeqStmts(statements, S),
            S,
            type=ir.FunctionType.Orchestration,
        )
    )
    before = program(functions)
    deduplicated = run(before)
    with passes.PassContext([]):
        expected = passes.convert_tensor_to_tile_ops()(before)
        after = passes.convert_tensor_to_tile_ops()(deduplicated)
    if grouped and not shared_symbol:
        # Preserving the pre-lowering Call.type was insufficient: the next pass
        # used the canonical callee's N for the second output that requires M.
        ir.assert_structural_equal(after, expected)
    expected_functions = 2 * (1 if shared_symbol else 2) + 1 if grouped else 2
    assert len(after.functions) == expected_functions
    dispatches = calls(after)
    assert len(dispatches) == 2
    for call, actual, dim in zip(dispatches, inputs, [n, m], strict=True):
        assert isinstance(call, ir.Submit if use_submit else ir.Call)
        assert call.args[0] is actual
        output_type = call.args[-1].type
        assert isinstance(output_type, ir.TensorType)
        assert output_type.shape[0] is dim
    assert dispatches[0].args[-1] is not dispatches[1].args[-1]


@pytest.mark.parametrize("shared_symbol", [False, True])
def test_group_member_view_symbols_keep_identity_even_with_equal_root_signatures(shared_symbol):
    n = var("extent", ir.ScalarType(D.INDEX))
    # Identical names are not evidence that the runtime extents are the same.
    m = n if shared_symbol else var("extent", ir.ScalarType(D.INDEX))
    functions = []
    for name, dim in [("a", n), ("b", m)]:
        typ = ir.TensorType(
            [16], D.FP32, tensor_view=ir.TensorView(layout=ir.TensorLayout.ND, valid_shape=[dim])
        )
        functions.extend([kernel(name, typ=typ), group("group_" + name, [name])])
    source = program([*functions, caller(["group_a", "group_b"])])
    result = run(source)
    assert len(result.functions) == (3 if shared_symbol else 5)


def group(name, members):
    x = var(name + "_x")
    stmts: list[ir.Stmt] = [ir.EvalStmt(ir.Call(ir.GlobalVar(m), [x], x.type, S), S) for m in members]
    stmts.append(ir.ReturnStmt([x], S))
    return ir.Function(name, [x], [x.type], ir.SeqStmts(stmts, S), S, type=ir.FunctionType.Group)


def test_exclusive_group_is_compared_and_removed_atomically():
    result = run(
        program(
            [
                kernel("ka"),
                kernel("kb"),
                group("ga", ["ka", "ka"]),
                group("gb", ["kb", "kb"]),
                caller(["ga", "gb"]),
            ]
        )
    )
    assert names(result) == {"ka", "ga", "main"}
    assert [c.op.name for c in calls(result)] == ["ga", "ga"]
    assert [c.op.name for c in calls(result, "ga")] == ["ka", "ka"]


def test_group_member_with_external_caller_is_never_merged_individually():
    result = run(
        program(
            [kernel("ka"), kernel("kb"), group("ga", ["ka"]), group("gb", ["kb"]), caller(["ga", "gb", "kb"])]
        )
    )
    assert names(result) == {"ka", "kb", "ga", "gb", "main"}


def test_group_member_alias_relationship_is_semantic():
    result = run(
        program(
            [
                kernel("ka"),
                kernel("kb"),
                kernel("kc"),
                group("ga", ["ka", "ka"]),
                group("gb", ["kb", "kc"]),
                caller(["ga", "gb"]),
            ]
        )
    )
    assert names(result) == {"ka", "kb", "kc", "ga", "gb", "main"}


def test_submit_keeps_dependencies_launch_spec_and_caller_type():
    x = var("x")
    dep = var("dep", ir.ScalarType(D.TASK_ID))
    tid = var("tid", dep.type)
    launch = ir.ConstInt(4, D.INDEX, S)
    submit = ir.Submit(
        ir.GlobalVar("b"),
        [x],
        [dep],
        {},
        {},
        tid.type,
        S,
        core_num=launch,
        sync_start=True,
        allow_early_resolve=True,
    )
    body = ir.SeqStmts(
        [
            ir.EvalStmt(ir.Call(ir.GlobalVar("a"), [x], x.type, S), S),
            ir.AssignStmt(tid, submit, S),
            ir.ReturnStmt([tid], S),
        ],
        S,
    )
    main = ir.Function("main", [x, dep], [tid.type], body, S, type=ir.FunctionType.Orchestration)
    result = run(program([kernel("a"), kernel("b"), main]))
    rewritten = calls(result)[1]
    assert isinstance(rewritten, ir.Submit)
    assert rewritten.op.name == "a"
    assert rewritten.args[0] is x and rewritten.deps[0] is dep
    assert rewritten.core_num is launch
    assert rewritten.type is submit.type
    assert rewritten.sync_start and rewritten.allow_early_resolve


def test_default_pipeline_reuses_kernel_but_keeps_both_output_allocations():
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910)

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def first(self, x: pl.Tensor[[16], pl.FP32]) -> pl.Tensor[[16], pl.FP32]:
            y = pl.add(x, x)
            return y

        @pl.function(type=pl.FunctionType.InCore)
        def second(self, z: pl.Tensor[[16], pl.FP32]) -> pl.Tensor[[16], pl.FP32]:
            result = pl.add(z, z)
            return result

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self, x: pl.Tensor[[16], pl.FP32]) -> pl.Tensor[[16], pl.FP32]:
            y = self.first(x)
            z = self.second(y)
            return z

    result = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(Before)
    kernels = [
        f for f in result.functions.values() if f.func_type in (ir.FunctionType.AIC, ir.FunctionType.AIV)
    ]
    assert len(kernels) == 1
    dispatches = calls(result)
    assert len(dispatches) == 2
    # Tensor-to-tile lowering creates a separate output per invocation.
    assert dispatches[0].args[-1] is not dispatches[1].args[-1]
    main = result.get_function("main")
    assert main is not None
    generated = codegen.generate_orchestration(result, main)
    assert len(generated.kernel_registrations) == 1
    assert generated.code.count("rt_submit_aiv_task(") == 2


def test_submit_preserves_omitted_runtime_output_and_predicate():
    typ = ir.TensorType([16], D.FP32)
    functions = []
    for name in ("a", "b"):
        output = var(name + "_out", typ)
        functions.append(
            ir.Function(
                name,
                [(output, ir.ParamDirection.Out)],
                [typ],
                ir.ReturnStmt([output], S),
                S,
                type=ir.FunctionType.InCore,
            )
        )
    predicate = ir.ConstBool(True, S)
    task_type = ir.ScalarType(D.TASK_ID)
    body = ir.SeqStmts(
        [
            ir.EvalStmt(ir.Submit(ir.GlobalVar(name), [], [], {}, {}, task_type, S, predicate=predicate), S)
            for name in ("a", "b")
        ],
        S,
    )
    functions.append(ir.Function("main", [], [], body, S, type=ir.FunctionType.Orchestration))
    result = run(program(functions))
    assert names(result) == {"a", "main"}
    for dispatch in calls(result):
        assert isinstance(dispatch, ir.Submit)
        assert dispatch.op.name == "a" and dispatch.args == []
        assert dispatch.predicate is predicate
        assert dispatch.type is task_type


@pytest.mark.parametrize("shared_symbol", [False, True])
@pytest.mark.parametrize("omit_output", [False, True])
def test_submit_runtime_outputs_keep_shape_distinct_callees(shared_symbol, omit_output):
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910)
    n = var("N", ir.ScalarType(D.INDEX))
    m = n if shared_symbol else var("M", ir.ScalarType(D.INDEX))
    functions, inputs, statements = [], [], []
    for name, dim in [("a", n), ("b", m)]:
        typ = ir.TensorType([dim], D.FP32)
        x, out = var(name + "_x", typ), var(name + "_out", typ)
        functions.append(
            ir.Function(
                name,
                [(x, ir.ParamDirection.In), (out, ir.ParamDirection.Out)],
                [typ],
                ir.ReturnStmt([out], S),
                S,
                type=ir.FunctionType.InCore,
            )
        )
        actuals = [var(name + "_input", typ)]
        if not omit_output:
            actuals.append(var(name + "_output", typ))
        inputs.extend(actuals)
        task_type = ir.TupleType([typ, ir.ScalarType(D.TASK_ID)])
        dispatch = ir.Submit(ir.GlobalVar(name), actuals, [], {}, {}, task_type, S)
        statements.append(ir.AssignStmt(var(name + "_result", task_type), dispatch, S))
    functions.append(
        ir.Function("main", inputs, [], ir.SeqStmts(statements, S), S, type=ir.FunctionType.Orchestration)
    )
    source = program(functions)
    after = run(source)
    can_merge = shared_symbol or not omit_output
    assert names(after) == ({"a", "main"} if can_merge else {"a", "b", "main"})
    with passes.PassContext([]):
        lowered = passes.convert_tensor_to_tile_ops()(after)
    for old, new in zip(calls(source), calls(lowered), strict=True):
        assert new.args == old.args and new.type is old.type
        if omit_output:
            callee = lowered.get_function(new.op.name)
            assert callee is not None
            # Runtime codegen reads the omitted shape directly from this type.
            ir.assert_structural_equal(callee.params[-1].type, old.args[0].type)


@pytest.mark.parametrize("left,right", [(0.0, -0.0), (1.0, 2.0)])
def test_float_bits_and_constants_are_not_interchangeable(left, right):
    functions = []
    for name, value in [("a", left), ("b", right)]:
        x = var(name)
        constant = ir.ConstFloat(value, D.FP32, S)
        functions.append(
            ir.Function(
                name, [x], [constant.type], ir.ReturnStmt([constant], S), S, type=ir.FunctionType.InCore
            )
        )
    assert names(run(program([*functions, caller(["a", "b"])]))) == {"a", "b", "main"}


def test_shared_expression_dag_is_not_expanded():
    functions = []
    for name in ["a", "b"]:
        x = var(name)
        expr = x
        for _ in range(80):
            expr = ir.Add(expr, expr, D.INT64, S)
        functions.append(
            ir.Function(name, [x], [x.type], ir.ReturnStmt([expr], S), S, type=ir.FunctionType.InCore)
        )
    assert names(run(program([*functions, caller(["a", "b"])]))) == {"a", "main"}


@pytest.mark.parametrize("depth, merges", [(80, True), (260, False)])
def test_shared_shape_dag_merges_within_admission_limit(depth, merges):
    """A shared shape DAG is linear to inspect; oversized signatures stay guarded."""
    functions, inputs, statements = [], [], []
    for name in ["a", "b"]:
        extent = var(name + "_n", ir.ScalarType(D.INDEX))
        for _ in range(depth):
            extent = ir.Add(extent, extent, D.INDEX, S)
        x = var(name + "_x", ir.TensorType([extent], D.FP32))
        functions.append(
            ir.Function(name, [x], [x.type], ir.ReturnStmt([x], S), S, type=ir.FunctionType.InCore)
        )
        inputs.append(x)
        statements.append(
            ir.AssignStmt(var(name + "_result", x.type), ir.Call(ir.GlobalVar(name), [x], x.type, S), S)
        )
    functions.append(
        ir.Function(
            "main",
            inputs,
            [],
            ir.SeqStmts([*statements, ir.ReturnStmt([], S)], S),
            S,
            type=ir.FunctionType.Orchestration,
        )
    )
    assert names(run(program(functions))) == ({"a", "main"} if merges else {"a", "b", "main"})


def test_bad_binding_is_skipped_without_certifying_it_valid():
    functions = []
    for name in ["a", "b"]:
        x, y, z = var(name), var("late"), var("early")
        body = ir.SeqStmts([ir.AssignStmt(z, y, S), ir.AssignStmt(y, x, S), ir.ReturnStmt([z], S)], S)
        functions.append(ir.Function(name, [x], [x.type], body, S, type=ir.FunctionType.InCore))
    assert names(run(program([*functions, caller(["a", "b"])]))) == {"a", "b", "main"}


def test_capture_hidden_in_return_type_is_rejected():
    hidden = var("hidden", ir.ScalarType(D.INDEX))
    functions = []
    for name in ["a", "b"]:
        x = var(name)
        functions.append(
            ir.Function(
                name,
                [x],
                [ir.TensorType([hidden], D.FP32)],
                ir.ReturnStmt([x], S),
                S,
                type=ir.FunctionType.InCore,
            )
        )
    assert names(run(program([*functions, caller(["a", "b"])]))) == {"a", "b", "main"}


@pytest.mark.parametrize("shared_with_body", [False, True])
def test_global_reference_in_metadata_pins_the_target(shared_with_body):
    main = caller(["a", "b"])
    metadata = (
        calls(program([main]))[1]
        if shared_with_body
        else ir.Call(ir.GlobalVar("b"), [main.params[0]], main.params[0].type, S)
    )
    main = ir.Function(
        main.name,
        main.params,
        main.return_types,
        main.body,
        S,
        type=main.func_type,
        attrs={"metadata_call": metadata},
    )
    assert names(run(program([kernel("a"), kernel("b"), main]))) == {"a", "b", "main"}


def test_metadata_reference_after_shared_body_node_pins_the_target():
    main = caller(["a", "b"])
    shared = calls(program([main]))[1]
    carrier = ir.Call(
        ir.GlobalVar("a"), [main.params[0]], {}, {"metadata_call": shared}, main.params[0].type, S
    )
    assert isinstance(main.body, ir.SeqStmts)
    body = ir.SeqStmts([*main.body.stmts[:-1], ir.EvalStmt(carrier, S), main.body.stmts[-1]], S)
    main = ir.Function(main.name, main.params, main.return_types, body, S, type=main.func_type)
    assert names(run(program([kernel("a"), kernel("b"), main]))) == {"a", "b", "main"}


def test_mixed_kernel_deduplication_precedes_aic_aiv_expansion():
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910)

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def first(
            self, a: pl.Tensor[[16, 16], pl.FP16], b: pl.Tensor[[16, 16], pl.FP16]
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            product = pl.matmul(a, b)
            result = pl.add(product, product)
            return result

        @pl.function(type=pl.FunctionType.InCore)
        def second(
            self, x: pl.Tensor[[16, 16], pl.FP16], y: pl.Tensor[[16, 16], pl.FP16]
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            product = pl.matmul(x, y)
            result = pl.add(product, product)
            return result

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self, a: pl.Tensor[[16, 16], pl.FP16], b: pl.Tensor[[16, 16], pl.FP16]
        ) -> tuple[pl.Tensor[[16, 16], pl.FP32], pl.Tensor[[16, 16], pl.FP32]]:
            first = self.first(a, b)
            second = self.second(a, b)
            return first, second

    result = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(Before)
    counts = [f.func_type for f in result.functions.values()]
    assert counts.count(ir.FunctionType.AIC) == 1
    assert counts.count(ir.FunctionType.AIV) == 1
    assert counts.count(ir.FunctionType.Group) == 1
    main = result.get_function("main")
    assert main is not None
    generated = codegen.generate_orchestration(result, main)
    assert len(generated.kernel_registrations) == 2
    assert generated.code.count("rt_submit_task(") == 2


def _deepseek_shape_case(case, name):
    """Reduced real signatures with writes and original-parameter return aliases."""
    dims = [var(f"extent_{i}", ir.ScalarType(D.INDEX)) for i in range(5)]
    n, m, b, r, c = dims
    if case == "ffn_norm":
        shapes = [[8, 4096], [1, 4096], [n, 4096], [n, 1], [8, 1], [n, 1]]
        outputs, returns = [2, 3, 4, 5], [2, 3, 4, 5]
    elif case == "hc_pre_linear_reduce":
        shapes = [[n * 4, 32], None, [m, 32]]
        outputs, returns = [2], [2]
    elif case == "kv_proj_matmul":
        shapes = [[n, 512], None, None, [m, 4096], [4096, 512]]
        outputs, returns = [0], [0]
    elif case == "csa_cache_writeback":
        shapes = [[n * 128, 512], [8], [8, 512]]
        outputs, returns = [0], []
    elif case == "score_topk_leaf":
        # Original argument slots 7, 8, 10, 13, 14, 15; B is shared by cache/scale.
        shapes = [[n, 64], [m], [b * 32, 128], [r, 1024], [b * 32, 1], [8, c]]
        outputs, returns = [3, 5], [3, 5]
    else:
        assert case == "kv_touch"
        shapes = [[n * 128, 512]]
        outputs, returns = [0], [0]
    params = [
        var(f"buffer_{i}", ir.TensorType(shape, D.FP32)) if shape else (n if i == 1 else m)
        for i, shape in enumerate(shapes)
    ]
    if case == "kv_proj_matmul":
        # Real projection families mix symbolic buffers with static NZ weights.
        params[4] = var(
            "weight", ir.TensorType([4096, 512], D.FP32, tensor_view=ir.TensorView(layout=ir.TensorLayout.NZ))
        )
    statements = []
    for i in outputs:
        typ = params[i].type
        assert isinstance(typ, ir.TensorType)
        # The computation is reduced; retain an observable write to each buffer.
        fill = tensor.full([1] * len(typ.shape), D.FP32, 0.0)
        tile = var(f"fill_{i}", fill.type)
        offsets: list[int | ir.Expr] = [0] * len(typ.shape)
        write = tensor.assemble(params[i], tile, offsets)
        statements.extend([ir.AssignStmt(tile, fill, S), ir.AssignStmt(var(f"written_{i}", typ), write, S)])
    statements.append(ir.ReturnStmt([params[i] for i in returns], S))
    directions = [
        ir.ParamDirection.InOut if i in outputs else ir.ParamDirection.In for i in range(len(params))
    ]
    func = ir.Function(
        name,
        list(zip(params, directions, strict=True)),
        [params[i].type for i in returns],
        ir.SeqStmts(statements, S),
        S,
        type=ir.FunctionType.InCore,
    )
    return func, returns


@pytest.mark.parametrize(
    "case",
    [
        "ffn_norm",
        "hc_pre_linear_reduce",
        "kv_proj_matmul",
        "csa_cache_writeback",
        "score_topk_leaf",
        "kv_touch",
    ],
)
@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("use_submit", [False, True])
def test_deepseek_shape_cases_preserve_buffers_and_dispatch(case, wrapped, use_submit):
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910)
    functions, inputs, statements, expected_args = [], [], [], []
    dependency = var("dependency", ir.ScalarType(D.TASK_ID))
    inputs.extend([dependency] if use_submit else [])
    for prefix in ["a", "b"]:
        func, returns = _deepseek_shape_case(case, prefix + "_" + case)
        functions.append(func)
        target = func.name
        types = list(func.return_types)
        result_type = types[0] if len(types) == 1 else ir.TupleType(types) if types else ir.UnknownType()
        if wrapped:
            forwarded = ir.Call(ir.GlobalVar(target), func.params, result_type, S)
            stmts: list[ir.Stmt] = (
                [ir.EvalStmt(forwarded, S)]
                if not types
                else [ir.AssignStmt(var("forwarded", result_type), forwarded, S)]
            )
            stmts.append(ir.ReturnStmt([func.params[i] for i in returns], S))
            target = "wrapper_" + prefix
            functions.append(
                ir.Function(
                    target,
                    list(zip(func.params, func.param_directions, strict=True)),
                    types,
                    ir.SeqStmts(stmts, S),
                    S,
                    type=ir.FunctionType.Spmd,
                )
            )
        actuals = [
            p if isinstance(p.type, ir.ScalarType) else var(prefix + p.name_hint, p.type) for p in func.params
        ]
        inputs.extend(actuals)
        expected_args.append(actuals)
        if use_submit:
            result_type = ir.TupleType([*types, ir.ScalarType(D.TASK_ID)])
            dispatch = ir.Submit(
                ir.GlobalVar(target),
                actuals,
                [dependency],
                {},
                {},
                result_type,
                S,
                core_num=ir.ConstInt(2, D.INDEX, S),
                allow_early_resolve=True,
                predicate=ir.ConstBool(True, S),
            )
        else:
            dispatch = ir.Call(ir.GlobalVar(target), actuals, result_type, S)
        statements.append(
            ir.AssignStmt(var(prefix + "result", result_type), dispatch, S)
            if types or use_submit
            else ir.EvalStmt(dispatch, S)
        )
    statements.append(ir.ReturnStmt([], S))
    functions.append(
        ir.Function("main", inputs, [], ir.SeqStmts(statements, S), S, type=ir.FunctionType.Orchestration)
    )
    source = program(functions)
    after = run(source)
    assert "b_" + case not in names(after)
    assert len(after.functions) == len(source.functions) - 1
    ir.assert_structural_equal(run(after), after)
    for old, new in zip(calls(source), calls(after), strict=True):
        assert new.args == old.args
        assert new.type is old.type
        if use_submit:
            assert isinstance(new, ir.Submit)
            assert new.deps == old.deps and new.core_num is old.core_num
            assert new.predicate is old.predicate and new.allow_early_resolve
    if wrapped:
        assert calls(after, "wrapper_b")[0].op.name == "a_" + case
    with passes.PassContext([]):
        lowered = passes.convert_tensor_to_tile_ops()(after)
    # These kernels already own their output buffers: no extra storage arguments.
    for call, actuals in zip(calls(lowered), expected_args, strict=True):
        assert call.args == actuals
    kernel_after = lowered.get_function("a_" + case)
    assert kernel_after is not None
    assert len(kernel_after.params) == len(functions[0].params)
    if wrapped:
        for prefix, actuals in zip(["a", "b"], expected_args, strict=True):
            call = calls(lowered, "wrapper_" + prefix)[0]
            actual_types = [actuals[i].type for i in returns]
            expected = actual_types[0] if len(actual_types) == 1 else ir.TupleType(actual_types)
            if actual_types:
                ir.assert_structural_equal(call.type, expected)
    else:
        for call, actuals in zip(calls(lowered), expected_args, strict=True):
            actual_types = [actuals[i].type for i in returns]
            if use_submit:
                actual_types.append(ir.ScalarType(D.TASK_ID))
            expected = (
                actual_types[0] if len(actual_types) == 1 and not use_submit else ir.TupleType(actual_types)
            )
            if actual_types:
                ir.assert_structural_equal(call.type, expected)


@pytest.mark.parametrize("tuple_result", [False, True])
@pytest.mark.parametrize("task_dtype", [D.TASK_ID, D.INT64])
@pytest.mark.parametrize("has_value_return", [False, True])
def test_shape_distinct_submit_checks_task_only_result(tuple_result, task_dtype, has_value_return):
    case = "kv_touch" if has_value_return else "csa_cache_writeback"
    task_type = ir.ScalarType(task_dtype)
    result_type = ir.TupleType([task_type]) if tuple_result else task_type
    functions, inputs, statements = [], [], []
    for name in ["a", "b"]:
        func, _ = _deepseek_shape_case(case, name)
        functions.append(func)
        actuals = [var(name + p.name_hint, p.type) for p in func.params]
        inputs.extend(actuals)
        dispatch = ir.Submit(ir.GlobalVar(name), actuals, [], {}, {}, result_type, S)
        statements.append(ir.AssignStmt(var(name + "task", result_type), dispatch, S))
    functions.append(
        ir.Function("main", inputs, [], ir.SeqStmts(statements, S), S, type=ir.FunctionType.Orchestration)
    )
    source = program(functions)
    after = run(source)
    can_merge = not has_value_return and task_dtype == D.TASK_ID
    assert names(after) == ({"a", "main"} if can_merge else {"a", "b", "main"})
    assert [call.op.name for call in calls(after)] == (["a", "a"] if can_merge else ["a", "b"])
    for old, new in zip(calls(source), calls(after), strict=True):
        assert new.args == old.args and new.type is old.type


@pytest.mark.parametrize("conflict", ["scalar", "repeated", "result"])
def test_shape_conflict_at_any_site_keeps_entire_candidate(conflict):
    n, m = var("N", ir.ScalarType(D.INDEX)), var("M", ir.ScalarType(D.INDEX))
    functions = []
    for name, dim in [("a", n), ("b", m)]:
        x, y = var("x", ir.TensorType([dim], D.FP32)), var("y", ir.TensorType([dim], D.FP32))
        functions.append(
            ir.Function(name, [x, y, dim], [x.type], ir.ReturnStmt([x], S), S, type=ir.FunctionType.InCore)
        )
    x = var("x", ir.TensorType([m], D.FP32))
    bad = var("bad", ir.TensorType([n], D.FP32))
    good_call = ir.Call(ir.GlobalVar("b"), [x, x, m], x.type, S)
    bad_call = ir.Call(
        ir.GlobalVar("b"),
        [x, bad if conflict == "repeated" else x, n if conflict == "scalar" else m],
        bad.type if conflict == "result" else x.type,
        S,
    )
    main = ir.Function(
        "main",
        [x, bad, m, n],
        [],
        ir.SeqStmts(
            [
                ir.EvalStmt(ir.Call(ir.GlobalVar("a"), [bad, bad, n], bad.type, S), S),
                ir.EvalStmt(good_call, S),
                ir.EvalStmt(bad_call, S),
                ir.ReturnStmt([], S),
            ],
            S,
        ),
        S,
        type=ir.FunctionType.Orchestration,
    )
    source = program([*functions, main])
    ir.assert_structural_equal(run(source), source)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
