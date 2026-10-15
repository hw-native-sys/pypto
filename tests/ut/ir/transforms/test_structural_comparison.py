# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Exact structural comparison and independently invoked binding verification."""

import random
import struct

import pytest
from pypto import DataType as D
from pypto import ir

from .storage_test_utils import without_call_memrefs

SPAN = ir.Span.unknown()


def var(name, dtype=D.INT64):
    return ir.Var(name, ir.ScalarType(dtype), SPAN)


def add(lhs, rhs):
    return ir.Add(lhs, rhs, D.INT64, SPAN)


def function(name, params, returns=None, attrs=None):
    returns = params if returns is None else returns
    return ir.Function(
        name, params, [v.type for v in returns], ir.ReturnStmt(returns, SPAN), SPAN, attrs=attrs
    )


def equal(lhs, rhs, expected=True, auto=False):
    assert ir.structural_equal(lhs, rhs, auto) is expected
    assert ir.structural_equal(rhs, lhs, auto) is expected
    if expected:
        ir.assert_structural_equal(lhs, rhs, auto)
        assert ir.structural_hash(lhs, auto) == ir.structural_hash(rhs, auto)
    else:
        with pytest.raises(ValueError, match="Structural mismatch at"):
            ir.assert_structural_equal(lhs, rhs, auto)


def call(attrs=None, op=None):
    return ir.Call(
        ir.Op("test.comparison") if op is None else op,
        [],
        {},
        attrs,
        ir.ScalarType(D.INT64),
        SPAN,
    )


def test_positional_and_keyword_arguments():
    x, y = var("x"), var("y")
    assert ir.structural_equal(x, y, True)
    assert ir.structural_hash(x, True) == ir.structural_hash(y, True)
    assert ir.structural_hash(type=x.type) == ir.structural_hash(x.type)
    assert ir.structural_hash(type=x.type) == ir.structural_hash(node=x.type)


@pytest.mark.parametrize("auto", [False, True])
def test_parameter_names_and_spans_are_ignored(auto):
    equal(function("f", [var("x")]), function("g", [var("y")]), auto=auto)


def test_free_variables_preserve_identity_even_with_same_name():
    x, y = var("x"), var("x")
    equal(x, x)
    equal(x, y, False)
    equal(x, y, auto=True)


def test_shared_variable_and_subtree_preserve_bijection():
    x, y = var("x"), var("y")
    equal(add(x, x), add(x, y), False, auto=True)
    shared = add(x, x)
    equal(add(shared, shared), add(add(y, y), shared), False, auto=True)
    equal(add(x, x), add(y, y), auto=True)


def test_parameter_and_external_reference_are_different():
    x, y = var("x"), var("y")
    equal(function("f", [x]), function("g", [y], [x]), False)


def test_return_alias_pattern_is_preserved():
    x, y, a, b = var("x"), var("y"), var("a"), var("b")
    equal(function("f", [x, y], [x, x]), function("g", [a, b], [a, b]), False)


def test_signature_shape_symbols_are_bound_bijectively():
    n, m, k = var("N", D.INDEX), var("M", D.INDEX), var("K", D.INDEX)

    def kernel(name, dims):
        params = [ir.Var(f"{name}{i}", ir.TensorType([d], D.FP32), SPAN) for i, d in enumerate(dims)]
        return function(name, params)

    equal(kernel("f", [n, n]), kernel("g", [m, m]))
    equal(kernel("f", [n, n]), kernel("g", [m, k]), False)


def test_tile_signature_shape_symbols_are_bound():
    n, m = var("n", D.INDEX), var("m", D.INDEX)
    lhs = function("f", [ir.Var("x", ir.TileType([n], D.FP32), SPAN)])
    rhs = function("g", [ir.Var("y", ir.TileType([m], D.FP32), SPAN)])
    equal(lhs, rhs)


def test_shape_can_reference_later_scalar_parameter():
    n, m = var("n", D.INDEX), var("m", D.INDEX)
    x = ir.Var("x", ir.TensorType([n], D.FP32), SPAN)
    y = ir.Var("y", ir.TensorType([m], D.FP32), SPAN)
    equal(function("f", [x, n]), function("g", [y, m]))


@pytest.mark.parametrize("dtype", [D.FP32, D.FP16])
def test_signed_zero_is_exact(dtype):
    lhs = ir.ConstFloat(0.0, dtype, SPAN)
    rhs = ir.ConstFloat(-0.0, dtype, SPAN)
    equal(lhs, rhs, False)


def test_nan_bits_are_reflexive_and_payload_sensitive():
    def nan(bits):
        return ir.ConstFloat(struct.unpack("d", struct.pack("Q", bits))[0], D.FP32, SPAN)

    lhs = nan(0x7FF8000000000001)
    equal(lhs, lhs)
    equal(lhs, nan(0x7FF8000000000001))
    equal(lhs, nan(0x7FF8000000000002), False)
    equal(lhs, nan(0xFFF8000000000001), False)


def test_integer_dtypes_are_exact():
    lhs = ir.ConstInt(1, D.INDEX, SPAN)
    rhs = ir.ConstInt(1, D.INT64, SPAN)
    equal(lhs, rhs, False)


def test_operator_kind_and_name_but_not_pointer_are_identity():
    equal(call(op=ir.Op("test.comparison")), call(op=ir.Op("test.comparison")))
    equal(call(op=ir.Op("test.comparison")), call(op=ir.GlobalVar("test.comparison")), False)
    equal(call(op=ir.GlobalVar("f")), call(op=ir.GlobalVar("g")), False)


def test_attrs_order_does_not_change_variable_mapping_or_hash():
    x, y = var("x"), var("y")
    lhs = call({"first": x, "second": add(x, x)})
    rhs = call({"second": add(y, y), "first": y})
    equal(lhs, rhs, auto=True)
    equal(lhs, call({"second": add(y, var("z")), "first": y}), False, auto=True)


@pytest.mark.parametrize(
    "left,right",
    [
        ({"flag": True}, {"flag": 1}),
        ({"number": 1.0}, {"number": 1}),
        ({"number": 0.0}, {"number": -0.0}),
        ({"cache_policy": [(0, 0)]}, {"cache_policy": [(0, 1)]}),
    ],
)
def test_attribute_type_and_value_are_exact(left, right):
    equal(call(left), call(right), False)


def test_duplicate_attribute_keys_are_unsupported():
    node = call([("flag", True), ("flag", True)])
    with pytest.raises(ir.StructuralComparisonUnsupported, match="DuplicateAttribute"):
        ir.structural_hash(node)
    with pytest.raises(ir.StructuralComparisonUnsupported):
        ir.structural_equal(node, node)


@pytest.mark.parametrize("location", ["type", "tuple", "parameter", "return", "local", "attr"])
def test_storage_is_supported_everywhere(location):
    base = ir.Var("base", ir.PtrType(), SPAN)
    tensor = ir.TensorType([16], D.FP32, ir.MemRef(base, 0, 64, SPAN))
    x = ir.Var("x", tensor, SPAN)
    node = {
        "type": tensor,
        "tuple": ir.TupleType([tensor]),
        "parameter": function("f", [x]),
        "return": function("f", [], [x]),
        "local": ir.AssignStmt(x, x, SPAN),
        "attr": call({"hidden": x}),
    }[location]
    equal(node, node)


def test_type_equality_includes_attached_memref():
    base = ir.Var("base", ir.PtrType(), SPAN)
    lhs = ir.TensorType([16], D.FP32, ir.MemRef(base, 0, 1024, SPAN))
    rhs = ir.TensorType([16], D.FP32, ir.MemRef(base, 512, 1024, SPAN))
    assert lhs != rhs
    equal(lhs, rhs, False)


def test_nullable_tile_view_and_layout():
    lhs = ir.TileType([16, 16], D.FP32, tile_view=ir.TileView())
    rhs = ir.TileType([16, 16], D.FP32, tile_view=ir.TileView())
    equal(lhs, rhs)
    equal(lhs, ir.TileType([16, 32], D.FP32), False)


def test_call_and_submit_are_different():
    result = ir.TupleType([ir.ScalarType(D.TASK_ID)])
    callee = ir.GlobalVar("kernel")
    lhs = ir.Call(callee, [], result, SPAN)
    rhs = ir.Submit(callee, [], [], result, SPAN)
    equal(lhs, rhs, False)


@pytest.mark.parametrize("option", ["deps", "core_num", "sync_start", "allow_early_resolve", "predicate"])
def test_submit_semantics_are_preserved(option):
    result = ir.TupleType([ir.ScalarType(D.TASK_ID)])
    dep = var("task", D.TASK_ID)

    def submit(changed=False):
        core_num = (
            ir.ConstInt(2, D.INDEX, SPAN)
            if option == "sync_start" or (changed and option == "core_num")
            else None
        )
        return ir.Submit(
            ir.GlobalVar("kernel"),
            [],
            [dep] if changed and option == "deps" else [],
            {},
            None,
            result,
            SPAN,
            core_num=core_num,
            sync_start=changed and option == "sync_start",
            allow_early_resolve=changed and option == "allow_early_resolve",
            predicate=ir.ConstBool(True, SPAN) if changed and option == "predicate" else None,
        )

    equal(submit(), submit(True), False)


def test_exponentially_shared_expression_dag():
    x, y = var("x"), var("y")
    lhs, rhs = x, y
    for _ in range(80):
        lhs = add(lhs, lhs)
        rhs = add(rhs, rhs)
    equal(lhs, rhs, auto=True)
    assert ir.structural_equal(lhs, rhs, True)


def test_pure_dag_sharing_is_not_identity():
    x, y = var("x"), var("y")
    shared = add(x, x)
    equal(add(shared, shared), add(add(y, y), add(y, y)), auto=True)


def test_mismatch_diagnostic_has_field_and_float_bits():
    with pytest.raises(ValueError, match=r"root.value.*0x"):
        ir.assert_structural_equal(ir.ConstFloat(0.0, D.FP32, SPAN), ir.ConstFloat(-0.0, D.FP32, SPAN))


def test_program_function_contexts_are_independent():
    x, y, z = var("x"), var("y"), var("z")
    lhs = ir.Program([function("a", [x]), function("b", [x])], "program", SPAN)
    rhs = ir.Program([function("a", [y]), function("b", [z])], "program", SPAN)
    equal(lhs, rhs)


@pytest.mark.parametrize("field", ["direction", "attrs", "kind", "runtime_binding", "dtype"])
def test_complete_function_contract(field):
    x = var("x")
    y = var("y", D.INT32 if field == "dtype" else D.INT64)
    lhs = function("f", [x])
    kwargs = {}
    params = [y]
    if field == "direction":
        params = [(y, ir.ParamDirection.InOut)]
    elif field == "attrs":
        kwargs["attrs"] = {"cache_policy": [(0, 1)]}
    elif field == "kind":
        kwargs["type"] = ir.FunctionType.InCore
    elif field == "runtime_binding":
        kwargs["requires_runtime_binding"] = True
    rhs = ir.Function("g", params, [y.type], ir.ReturnStmt([y], SPAN), SPAN, **kwargs)
    equal(lhs, rhs, False)


def test_iter_arg_initializer_is_an_outer_reference():
    x, y = var("x"), var("y")

    def loop(seed, name):
        i = var(name, D.INDEX)
        carry = ir.IterArg(name + "_carry", seed.type, seed, SPAN)
        out = var(name + "_out")
        return ir.ForStmt(
            i,
            ir.ConstInt(0, D.INDEX, SPAN),
            ir.ConstInt(4, D.INDEX, SPAN),
            ir.ConstInt(1, D.INDEX, SPAN),
            [carry],
            ir.YieldStmt([carry], SPAN),
            [out],
            SPAN,
        )

    equal(loop(x, "i"), loop(x, "j"))
    equal(loop(x, "i"), loop(y, "j"), False)
    equal(loop(x, "i"), loop(y, "j"), auto=True)


def test_valid_shape_symbols_need_runtime_readable_sources():
    def kernel(name, compound=False):
        n = var(name, D.INDEX)
        extent = ir.Mul(n, ir.ConstInt(2, D.INDEX, SPAN), D.INDEX, SPAN) if compound else n
        view = ir.TensorView(stride=[1], layout=ir.TensorLayout.ND, valid_shape=[extent])
        x = ir.Var(name + "_tensor", ir.TensorType([16], D.FP32, tensor_view=view), SPAN)
        return function(name, [x])

    equal(kernel("f"), kernel("g"))
    with pytest.raises(ir.StructuralComparisonUnsupported, match="UnboundValidShapeSymbol"):
        ir.verify_structural_bindings(kernel("bad", True))


def test_repeated_definition_is_unsupported():
    x = var("x")
    body = ir.SeqStmts(
        [
            ir.AssignStmt(x, ir.ConstInt(1, D.INT64, SPAN), SPAN),
            ir.AssignStmt(x, ir.ConstInt(2, D.INT64, SPAN), SPAN),
        ],
        SPAN,
    )
    with pytest.raises(ir.StructuralComparisonUnsupported, match="RepeatedDefinition"):
        ir.verify_structural_bindings(body)


def test_unresolved_types_are_structurally_comparable():
    equal(ir.UnknownType(), ir.UnknownType())


def test_depth_limit_is_explicit_instead_of_stack_overflow():
    expr = var("x")
    for _ in range(600):
        expr = add(expr, ir.ConstInt(1, D.INT64, SPAN))
    with pytest.raises(ir.StructuralComparisonUnsupported, match="DepthLimitExceeded"):
        ir.structural_hash(expr)


def test_signature_depth_limit_is_explicit():
    extent = var("N", D.INDEX)
    for _ in range(600):
        extent = ir.Add(extent, ir.ConstInt(1, D.INDEX, SPAN), D.INDEX, SPAN)
    node = function("f", [ir.Var("x", ir.TensorType([extent], D.FP32), SPAN)])
    with pytest.raises(ir.StructuralComparisonUnsupported, match="DepthLimitExceeded"):
        ir.structural_hash(node)


def test_shared_statement_is_unsupported():
    stmt = ir.ReturnStmt([], SPAN)
    with pytest.raises(ir.StructuralComparisonUnsupported, match="SharedStatement"):
        ir.verify_structural_bindings(ir.SeqStmts([stmt, stmt], SPAN))


@pytest.mark.parametrize("seed", range(10))
def test_generated_graphs_preserve_equivalence_under_renaming_and_sharing(seed):
    rng = random.Random(seed)
    edges = [(rng.randrange(i), rng.randrange(i)) for i in range(3, 35)]

    def build(name, share):
        params = [var(f"{name}{i}") for i in range(3)]
        cache = dict(enumerate(params))

        def node(index):
            if index in cache:
                return cache[index]
            left, right = edges[index - 3]
            result = add(node(left), node(right))
            if share:
                cache[index] = result
            return result

        result = node(34)
        return ir.Function(name, params, [result.type], ir.ReturnStmt([result], SPAN), SPAN)

    a, b, c = build("a", True), build("b", False), build("c", True)
    equal(a, a)
    equal(a, b)
    equal(b, c)
    equal(a, c)


def assert_unsupported(node, reason="BindingNotVisible"):
    for operation in (lambda: ir.verify_structural_bindings(node),):
        with pytest.raises(ir.StructuralComparisonUnsupported) as exc:
            operation()
        assert exc.value.reason == reason
        assert exc.value.field_path.startswith("root")


@pytest.mark.parametrize("location", ["rhs", "before", "type", "attr", "shared"])
def test_local_binder_must_be_visible_at_every_use(location):
    x = var("x")
    one = ir.ConstInt(1, D.INT64, SPAN)
    shared = add(x, one)
    if location == "rhs":
        body = ir.AssignStmt(x, shared, SPAN)
    else:
        use = {
            "before": x,
            "type": ir.Var("tensor", ir.TensorType([x], D.FP32), SPAN),
            "attr": call({"capture": x}),
            "shared": shared,
        }[location]
        body = ir.SeqStmts(
            [ir.EvalStmt(use, SPAN), ir.AssignStmt(x, one, SPAN), ir.EvalStmt(shared, SPAN)], SPAN
        )
    assert_unsupported(body)


@pytest.mark.parametrize("escape", ["direct", "shared", "type", "attr", "sibling"])
def test_branch_binder_cannot_escape_even_through_cached_dag(escape):
    x = var("x")
    use = {
        "direct": x,
        "shared": add(x, x),
        "type": ir.Var("tensor", ir.TensorType([x], D.FP32), SPAN),
        "attr": call({"hidden": x}),
        "sibling": add(x, x),
    }[escape]
    then = ir.SeqStmts([ir.AssignStmt(x, ir.ConstInt(1, D.INT64, SPAN), SPAN), ir.EvalStmt(use, SPAN)], SPAN)
    branch = ir.IfStmt(
        ir.ConstBool(True, SPAN), then, ir.EvalStmt(use, SPAN) if escape == "sibling" else None, [], SPAN
    )
    assert_unsupported(ir.SeqStmts([branch, ir.EvalStmt(use, SPAN)], SPAN))


def test_if_results_bind_after_both_branches_and_shadow_names_are_irrelevant():
    def build(prefix):
        seed, local, result = var(prefix), var(prefix), var(prefix)
        branch = ir.IfStmt(
            ir.ConstBool(True, SPAN),
            ir.SeqStmts([ir.AssignStmt(local, seed, SPAN), ir.YieldStmt([local], SPAN)], SPAN),
            ir.YieldStmt([seed], SPAN),
            [result],
            SPAN,
        )
        body = ir.SeqStmts([branch, ir.ReturnStmt([result, seed], SPAN)], SPAN)
        return ir.Function(prefix, [seed], [seed.type, seed.type], body, SPAN)

    equal(build("same"), build("renamed"))
    result = var("result")
    assert_unsupported(
        ir.IfStmt(ir.ConstBool(True, SPAN), ir.YieldStmt([result], SPAN), None, [result], SPAN)
    )


@pytest.mark.parametrize("loop_kind", ["for", "while"])
@pytest.mark.parametrize("invalid", [None, "escape", "init", "result", "local"])
def test_loop_binding_lifetimes(loop_kind, invalid):
    def build(prefix):
        seed, output, local = var(prefix), var(prefix), var(prefix)
        outputs = [var(prefix), output]
        first = ir.IterArg(prefix, seed.type, seed, SPAN)
        second = ir.IterArg(prefix, seed.type, first if invalid == "init" else seed, SPAN)
        yielded = output if invalid == "result" else second
        body = ir.SeqStmts([ir.AssignStmt(local, first, SPAN), ir.YieldStmt([first, yielded], SPAN)], SPAN)
        if loop_kind == "for":
            loop = ir.ForStmt(
                var(prefix, D.INDEX),
                ir.ConstInt(0, D.INDEX, SPAN),
                ir.ConstInt(4, D.INDEX, SPAN),
                ir.ConstInt(1, D.INDEX, SPAN),
                [first, second],
                body,
                outputs,
                SPAN,
            )
        else:
            loop = ir.WhileStmt(ir.Lt(first, second, D.BOOL, SPAN), [first, second], body, outputs, SPAN)
        returned = first if invalid == "escape" else local if invalid == "local" else output
        return ir.Function(
            prefix, [seed], [seed.type], ir.SeqStmts([loop, ir.ReturnStmt([returned], SPAN)], SPAN), SPAN
        )

    if invalid:
        assert_unsupported(build("x"))
    else:
        equal(build("x"), build("y"))


def test_signature_metadata_does_not_capture_a_later_local_definition():
    local = var("n", D.INDEX)
    tensor = ir.Var(
        "t",
        ir.TensorType([16], D.FP32, tensor_view=ir.TensorView(stride=[local], layout=ir.TensorLayout.ND)),
        SPAN,
    )
    body = ir.AssignStmt(local, ir.ConstInt(1, D.INDEX, SPAN), SPAN)
    assert_unsupported(ir.Function("f", [tensor], [], body, SPAN))


def test_compound_valid_shape_can_use_a_later_bare_slot():
    def build(prefix):
        n = var(prefix, D.INDEX)
        params = [
            ir.Var(
                prefix + str(i),
                ir.TensorType(
                    [16],
                    D.FP32,
                    tensor_view=ir.TensorView(stride=[1], valid_shape=[extent], layout=ir.TensorLayout.ND),
                ),
                SPAN,
            )
            for i, extent in enumerate([ir.Mul(n, ir.ConstInt(2, D.INDEX, SPAN), D.INDEX, SPAN), n])
        ]
        return function(prefix, params)

    equal(build("n"), build("m"))


@pytest.mark.parametrize(
    "factory",
    [
        ir.VoidType,
        ir.PrefetchAsyncContextType,
        ir.AsyncEventType,
        ir.AsyncSessionType,
        lambda: ir.ScalarType(D.FP32),
        lambda: ir.TensorType([16], D.FP32),
        lambda: ir.DistributedTensorType([16], D.FP32),
        lambda: ir.TileType([16, 16], D.FP32),
        lambda: ir.ArrayType(D.INT32, 16),
        lambda: ir.BufferType([16, 16], D.FP32, ir.MemorySpace.Mat),
        lambda: ir.MultiBufferType(ir.BufferType([16, 16], D.FP32, ir.MemorySpace.Mat), 2),
        lambda: ir.TupleType([ir.ScalarType(D.INT32), ir.TileType([16, 16], D.FP32)]),
    ],
)
def test_supported_type_projection_parity(factory):
    equal(factory(), factory())


@pytest.mark.parametrize(
    "field,changed",
    [
        ("valid_shape", [8, 16]),
        ("stride", [32, 1]),
        ("start_offset", 0),
        ("blayout", ir.TileLayout.col_major),
        ("slayout", ir.TileLayout.row_major),
        ("fractal", 1024),
        ("pad", ir.PadValue.zero),
    ],
)
def test_tile_view_single_field_changes(field, changed):
    lhs = ir.TileType([16, 16], D.FP32, tile_view=ir.TileView())
    rhs = ir.TileType([16, 16], D.FP32, tile_view=ir.TileView(**{field: changed}))
    equal(lhs, rhs, False)


def test_many_uses_of_large_shared_type():
    def build(prefix):
        n = var(prefix, D.INDEX)
        shape = n
        for _ in range(60):
            shape = ir.Add(shape, shape, D.INDEX, SPAN)
        tensor = ir.TensorType([shape], D.FP32)
        params = [ir.Var(f"{prefix}{i}", tensor, SPAN) for i in range(100)]
        return function(prefix, params)

    equal(build("n"), build("m"))


@pytest.mark.parametrize(
    "type_factory",
    [
        lambda: ir.MemRef(ir.Var("base", ir.PtrType(), SPAN), 0, 64, SPAN).type,
        ir.PtrType,
        ir.WindowBufferType,
        ir.CommCtxType,
    ],
)
def test_resource_marker_types_are_supported(type_factory):
    node = type_factory()
    equal(node, type_factory())


@pytest.mark.parametrize("attached", [False, True])
def test_window_storage_is_supported(attached):
    base = ir.Var("base", ir.PtrType(), SPAN)
    window = ir.WindowBuffer(base, ir.ConstInt(64, D.INDEX, SPAN), span=SPAN)
    if attached:
        node = ir.Var(
            "slice", ir.DistributedTensorType([ir.ConstInt(16, D.INDEX, SPAN)], D.FP32, window), SPAN
        )
    else:
        node = window
    equal(node, node)


def test_open_stride_symbol_is_not_implicitly_bound():
    def build(prefix):
        n = var(prefix, D.INDEX)
        tensor = ir.TensorType([16], D.FP32, tensor_view=ir.TensorView(stride=[n], layout=ir.TensorLayout.ND))
        return function(prefix, [ir.Var(prefix + "_tensor", tensor, SPAN)])

    equal(build("n"), build("m"), False)
    equal(build("n"), build("m"), auto=True)


def test_hash_revisits_subtree_after_a_binding_is_introduced():
    x, y = var("x"), var("y")
    shared = add(x, x)
    one = ir.ConstInt(1, D.INT64, SPAN)
    lhs = ir.SeqStmts(
        [ir.EvalStmt(shared, SPAN), ir.AssignStmt(x, one, SPAN), ir.EvalStmt(shared, SPAN)], SPAN
    )
    rhs = ir.SeqStmts(
        [ir.EvalStmt(shared, SPAN), ir.AssignStmt(y, one, SPAN), ir.EvalStmt(add(y, y), SPAN)], SPAN
    )
    equal(lhs, rhs, False)
    assert_unsupported(lhs)


@pytest.mark.parametrize("placement", ["standalone", "sequence", "nested", "function"])
@pytest.mark.parametrize("kind", ["incore", "spmd"])
def test_scope_body_is_checked_once_and_definitions_flow_out(placement, kind):
    def build(name):
        x = var(name)
        body = ir.AssignStmt(x, ir.ConstInt(1, D.INT64, SPAN), SPAN)
        scope = (
            ir.InCoreScopeStmt(body=body, span=SPAN)
            if kind == "incore"
            else ir.SpmdScopeStmt(4, body=body, span=SPAN)
        )
        if placement == "standalone":
            return scope
        if placement == "nested":
            scope = ir.InCoreScopeStmt(body=scope, span=SPAN)
        result = ir.SeqStmts([scope, ir.ReturnStmt([x], SPAN)], SPAN)
        if placement == "function":
            return ir.Function(name, [], [x.type], result, SPAN)
        return result

    equal(build("x"), build("y"))


@pytest.mark.parametrize("placement", ["standalone", "sequence", "nested", "function"])
def test_scope_metadata_cannot_reference_body_local_binding(placement):
    x = var("x")
    body = ir.AssignStmt(x, ir.ConstInt(4, D.INT64, SPAN), SPAN)
    scope = ir.SpmdScopeStmt(x, body=body, span=SPAN)
    if placement == "sequence":
        root = ir.SeqStmts([scope], SPAN)
    elif placement == "nested":
        root = ir.InCoreScopeStmt(body=scope, span=SPAN)
    elif placement == "function":
        root = ir.Function("f", [], [], scope, SPAN)
    else:
        root = scope
    assert_unsupported(root)


def test_scope_metadata_can_reference_function_parameter():
    def build(name):
        count = var(name)
        scope = ir.SpmdScopeStmt(count, body=ir.ReturnStmt([], SPAN), span=SPAN)
        return ir.Function(name, [count], [], scope, SPAN)

    equal(build("x"), build("y"))


@pytest.mark.parametrize("location", ["type", "attribute"])
def test_unknown_dtype_codes_remain_distinct_after_deserialization(location):
    payload = ir.serialize(var("x"))
    encoded_dtype = b"\xa5dtype" + bytes([D.INT64.code()])
    assert payload.count(encoded_dtype) == 1

    def decoded(code):
        restored = ir.deserialize(payload.replace(encoded_dtype, b"\xa5dtype" + bytes([code])))
        assert isinstance(restored, ir.Var)
        assert isinstance(restored.type, ir.ScalarType)
        assert restored.type.dtype.code() == code
        assert restored.type.dtype.to_string() == "unknown"
        return restored.type if location == "type" else call({"dtype": restored.type.dtype})

    equal(decoded(0x60), decoded(0x60))
    equal(decoded(0x60), decoded(0x61), False)


def test_hash_invalidates_cached_ancestors_and_type_uses():
    x, y = var("x"), var("y")
    extent = add(x, x)
    shared = ir.Var("tensor", ir.TensorType([extent], D.FP32), SPAN)
    # A Var's type is visited in a definition field. The expression reference
    # shares the same extent DAG before and after the scalar becomes bound.
    before = ir.EvalStmt(add(extent, extent), SPAN)
    lhs = ir.SeqStmts(
        [before, ir.AssignStmt(x, ir.ConstInt(1, D.INT64, SPAN), SPAN), ir.AssignStmt(shared, call(), SPAN)],
        SPAN,
    )
    other = ir.Var("other", ir.TensorType([add(y, y)], D.FP32), SPAN)
    rhs = ir.SeqStmts(
        [before, ir.AssignStmt(y, ir.ConstInt(1, D.INT64, SPAN), SPAN), ir.AssignStmt(other, call(), SPAN)],
        SPAN,
    )
    equal(lhs, rhs, False)
    assert_unsupported(lhs)


@pytest.mark.parametrize("nested", [False, True])
def test_signature_symbols_inside_tuple_constructors(nested):
    def kernel(name, repeated=True):
        n, m = var(name, D.INDEX), var(name, D.INDEX)
        pair = ir.MakeTuple([n, n if repeated else m], SPAN)
        if nested:
            pair = ir.TupleGetItemExpr(ir.MakeTuple([pair], SPAN), 0, SPAN)
        extent = ir.TupleGetItemExpr(pair, 0, SPAN)
        tensor = ir.Var(name, ir.TensorType([extent], D.FP32), SPAN)
        return function(name, [tensor])

    equal(kernel("left"), kernel("right"))
    equal(kernel("left"), kernel("right", repeated=False), False)


def test_unbound_valid_shape_symbol_inside_tuple_is_rejected():
    extent = ir.TupleGetItemExpr(ir.MakeTuple([var("n", D.INDEX)], SPAN), 0, SPAN)
    tensor = ir.Var(
        "tensor",
        ir.TensorType(
            [16], D.FP32, tensor_view=ir.TensorView(valid_shape=[extent], layout=ir.TensorLayout.ND)
        ),
        SPAN,
    )
    assert_unsupported(function("kernel", [tensor]), "UnboundValidShapeSymbol")


@pytest.mark.parametrize("scope", ["branch", "for", "while"])
@pytest.mark.parametrize("location", ["direct", "shared", "type"])
@pytest.mark.parametrize("before", [False, True])
def test_visibility_error_reports_invalid_use_endpoint(scope, location, before):
    x = var("local")
    shared = add(add(x, x), x)
    use = {
        "direct": x,
        "shared": shared,
        "type": ir.Var("tensor", ir.TensorType([shared], D.FP32), SPAN),
    }[location]
    body = ir.SeqStmts([ir.AssignStmt(x, ir.ConstInt(1, D.INT64, SPAN), SPAN), ir.EvalStmt(use, SPAN)], SPAN)
    if scope == "branch":
        region = ir.IfStmt(ir.ConstBool(True, SPAN), body, None, [], SPAN)
    elif scope == "for":
        region = ir.ForStmt(
            var("i", D.INDEX),
            ir.ConstInt(0, D.INDEX, SPAN),
            ir.ConstInt(1, D.INDEX, SPAN),
            ir.ConstInt(1, D.INDEX, SPAN),
            [],
            body,
            [],
            SPAN,
        )
    else:
        region = ir.WhileStmt(ir.ConstBool(True, SPAN), [], body, [], SPAN)
    outside = ir.EvalStmt(use, SPAN)
    root = ir.SeqStmts([outside, region] if before else [region, outside], SPAN)
    prefix = f"root.stmts.[{0 if before else 1}].expr"
    if location == "type":
        prefix += ".type.shape.[0]"
    suffixes = [""] if location == "direct" else [".left.left", ".left.right", ".right"]
    for operation in (lambda: ir.verify_structural_bindings(root),):
        with pytest.raises(ir.StructuralComparisonUnsupported) as exc:
            operation()
        assert exc.value.reason == "BindingNotVisible"
        assert exc.value.field_path in {prefix + suffix for suffix in suffixes}


@pytest.mark.parametrize("shared_allocation", [False, True])
def test_function_storage_aliases_survive_renaming_and_serialization(shared_allocation):
    def build(prefix, share):
        a = ir.Var(prefix + "a", ir.PtrType(), SPAN)
        b = a if share else ir.Var(prefix + "b", ir.PtrType(), SPAN)
        params = [
            ir.Var(prefix + str(i), ir.TensorType([16], D.FP32, ir.MemRef(base, 0, 64, SPAN)), SPAN)
            for i, base in enumerate([a, b])
        ]
        return function(prefix, params)

    lhs = build("left", shared_allocation)
    equal(lhs, build("right", shared_allocation))
    equal(lhs, build("different", not shared_allocation), False)
    equal(lhs, ir.deserialize(ir.serialize(lhs)))


def test_memref_descriptor_sharing_does_not_change_allocation_aliases():
    base = ir.Var("base", ir.PtrType(), SPAN)
    first = ir.MemRef(base, 0, 64, SPAN)
    second = ir.MemRef(base, 0, 64, SPAN)
    equal(first, second)
    equal(first, ir.MemRef(base, 16, 64, SPAN), False)
    other_base = ir.Var("base", ir.PtrType(), SPAN)
    equal(first, ir.MemRef(other_base, 0, 64, SPAN), False)
    equal(first, ir.MemRef(other_base, 0, 64, SPAN), auto=True)


def test_comparison_does_not_implicitly_verify_binding_validity():
    x = var("x")
    malformed = ir.SeqStmts(
        [ir.EvalStmt(x, SPAN), ir.AssignStmt(x, ir.ConstInt(1, D.INT64, SPAN), SPAN)], SPAN
    )
    equal(malformed, malformed)
    with pytest.raises(ir.StructuralComparisonUnsupported, match="BindingNotVisible"):
        ir.verify_structural_bindings(malformed)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


def test_storage_test_normalization_keeps_variable_allocations():
    """Normalizing inferred Call metadata must not hide storage pass regressions."""

    def program(offset, call_memref):
        base = ir.Var("base", ir.PtrType(), SPAN)
        memref = ir.MemRef(base, offset, 16, SPAN)
        shape = [ir.ConstInt(4, D.INDEX, SPAN)]
        type_ = ir.TensorType(shape, D.FP32, memref)
        result = ir.Var("result", type_, SPAN)
        call_type = type_ if call_memref else ir.TensorType(shape, D.FP32)
        call = ir.Call(ir.GlobalVar("callee"), [], call_type, SPAN)
        body = ir.SeqStmts([ir.AssignStmt(result, call, SPAN), ir.ReturnStmt([result], SPAN)], SPAN)
        func = ir.Function("main", [], [], body, SPAN)
        return ir.Program([func], "Example", SPAN)

    lhs, rhs = program(0, False), program(0, True)
    equal(lhs, rhs, False)
    equal(without_call_memrefs(lhs), without_call_memrefs(rhs))
    equal(without_call_memrefs(lhs), without_call_memrefs(program(16, True)), False)
