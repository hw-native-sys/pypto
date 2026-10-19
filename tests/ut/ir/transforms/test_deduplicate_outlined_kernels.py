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
from pypto.ir.op import tensor
from pypto.ir.pass_manager import OptimizationStrategy, PassManager

S = ir.Span.unknown()


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


def test_alpha_equivalent_signatures_with_distinct_symbols_are_not_aliased():
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
    assert names(result) == {"a", "b", "c", "main"}
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
    if not shared_symbol:
        # Preserving the pre-lowering Call.type was insufficient: the next pass
        # used the canonical callee's N for the second output that requires M.
        ir.assert_structural_equal(after, expected)
    expected_functions = (2 if grouped else 1) * (1 if shared_symbol else 2) + 1
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
