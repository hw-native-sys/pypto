# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Exact structural comparison for kernel deduplication candidates."""

import random
import struct

import pytest
from pypto import DataType as D
from pypto import ir

STRICT = ir.StructuralComparisonMode.Strict
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
    assert ir.structural_equal(lhs, rhs, auto, mode=STRICT) is expected
    assert ir.structural_equal(rhs, lhs, auto, mode=STRICT) is expected
    if expected:
        ir.assert_structural_equal(lhs, rhs, auto, mode=STRICT)
        assert ir.structural_hash(lhs, auto, mode=STRICT) == ir.structural_hash(rhs, auto, mode=STRICT)
    else:
        with pytest.raises(ValueError, match="Strict structural mismatch at"):
            ir.assert_structural_equal(lhs, rhs, auto, mode=STRICT)


def call(attrs=None, op=None):
    return ir.Call(
        ir.Op("test.comparison") if op is None else op,
        [],
        {},
        attrs,
        ir.ScalarType(D.INT64),
        SPAN,
    )


def test_old_positional_arguments_and_default_mode():
    x, y = var("x"), var("y")
    assert ir.structural_equal(x, y, True)
    assert ir.structural_equal(x, y, True, mode=ir.StructuralComparisonMode.Default)
    assert ir.structural_hash(x, True) == ir.structural_hash(y, True)
    assert ir.structural_hash(type=x.type) == ir.structural_hash(x.type)
    assert ir.structural_hash(type=x.type, mode=STRICT) == ir.structural_hash(node=x.type, mode=STRICT)


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


def test_shape_can_reference_later_scalar_parameter():
    n, m = var("n", D.INDEX), var("m", D.INDEX)
    x = ir.Var("x", ir.TensorType([n], D.FP32), SPAN)
    y = ir.Var("y", ir.TensorType([m], D.FP32), SPAN)
    equal(function("f", [x, n]), function("g", [y, m]))


@pytest.mark.parametrize("dtype", [D.FP32, D.FP16])
def test_signed_zero_is_exact(dtype):
    lhs = ir.ConstFloat(0.0, dtype, SPAN)
    rhs = ir.ConstFloat(-0.0, dtype, SPAN)
    assert ir.structural_equal(lhs, rhs)
    equal(lhs, rhs, False)


def test_nan_bits_are_reflexive_and_payload_sensitive():
    def nan(bits):
        return ir.ConstFloat(struct.unpack("d", struct.pack("Q", bits))[0], D.FP32, SPAN)

    lhs = nan(0x7FF8000000000001)
    equal(lhs, lhs)
    equal(lhs, nan(0x7FF8000000000001))
    equal(lhs, nan(0x7FF8000000000002), False)
    equal(lhs, nan(0xFFF8000000000001), False)


def test_integer_syntax_relaxation_is_default_only():
    lhs = ir.ConstInt(1, D.INDEX, SPAN)
    rhs = ir.ConstInt(1, D.INT64, SPAN)
    assert ir.structural_equal(lhs, rhs)
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
        ir.structural_hash(node, mode=STRICT)
    with pytest.raises(ir.StructuralComparisonUnsupported):
        ir.structural_equal(node, node, mode=STRICT)


@pytest.mark.parametrize("location", ["type", "tuple", "parameter", "return", "local", "attr"])
def test_storage_is_rejected_everywhere(location):
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
    with pytest.raises(ir.StructuralComparisonUnsupported) as exc:
        ir.structural_hash(node, mode=STRICT)
    assert exc.value.reason == "UnsupportedStorage"
    assert "memref" in exc.value.field_path
    for compare in (ir.structural_equal, ir.assert_structural_equal):
        with pytest.raises(ir.StructuralComparisonUnsupported):
            if isinstance(node, ir.Type):
                compare(node, node, mode=STRICT)
            else:
                compare(node, node, mode=STRICT)


def test_default_type_equality_still_ignores_attached_memref():
    base = ir.Var("base", ir.PtrType(), SPAN)
    lhs = ir.TensorType([16], D.FP32, ir.MemRef(base, 0, 1024, SPAN))
    rhs = ir.TensorType([16], D.FP32, ir.MemRef(base, 512, 1024, SPAN))
    assert lhs == rhs
    assert hash(lhs) == hash(rhs)


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
        ir.assert_structural_equal(
            ir.ConstFloat(0.0, D.FP32, SPAN), ir.ConstFloat(-0.0, D.FP32, SPAN), mode=STRICT
        )


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
        ir.structural_hash(kernel("bad", True), mode=STRICT)


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
        ir.structural_hash(body, mode=STRICT)


def test_unresolved_types_are_unsupported():
    with pytest.raises(ir.StructuralComparisonUnsupported, match="UnresolvedType"):
        ir.structural_equal(ir.UnknownType(), ir.UnknownType(), mode=STRICT)


def test_depth_limit_is_explicit_instead_of_stack_overflow():
    expr = var("x")
    for _ in range(600):
        expr = add(expr, ir.ConstInt(1, D.INT64, SPAN))
    with pytest.raises(ir.StructuralComparisonUnsupported, match="DepthLimitExceeded"):
        ir.structural_hash(expr, mode=STRICT)


def test_signature_depth_limit_is_explicit():
    extent = var("N", D.INDEX)
    for _ in range(600):
        extent = ir.Add(extent, ir.ConstInt(1, D.INDEX, SPAN), D.INDEX, SPAN)
    node = function("f", [ir.Var("x", ir.TensorType([extent], D.FP32), SPAN)])
    with pytest.raises(ir.StructuralComparisonUnsupported, match="DepthLimitExceeded"):
        ir.structural_hash(node, mode=STRICT)


def test_shared_statement_is_unsupported():
    stmt = ir.ReturnStmt([], SPAN)
    with pytest.raises(ir.StructuralComparisonUnsupported, match="SharedStatement"):
        ir.structural_hash(ir.SeqStmts([stmt, stmt], SPAN), mode=STRICT)


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
    for operation in (
        lambda: ir.structural_hash(node, mode=STRICT),
        lambda: ir.structural_equal(node, node, mode=STRICT),
        lambda: ir.assert_structural_equal(node, node, mode=STRICT),
    ):
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
def test_resource_marker_types_are_unsupported(type_factory):
    node = type_factory()
    with pytest.raises(ir.StructuralComparisonUnsupported, match="UnsupportedStorage"):
        ir.structural_hash(node, mode=STRICT)


@pytest.mark.parametrize("attached", [False, True])
def test_window_storage_is_unsupported(attached):
    base = ir.Var("base", ir.PtrType(), SPAN)
    window = ir.WindowBuffer(base, ir.ConstInt(64, D.INDEX, SPAN), span=SPAN)
    if attached:
        node = ir.Var(
            "slice", ir.DistributedTensorType([ir.ConstInt(16, D.INDEX, SPAN)], D.FP32, window), SPAN
        )
    else:
        node = window
    assert_unsupported(node, "UnsupportedStorage")


def test_open_stride_symbol_is_not_implicitly_bound():
    def build(prefix):
        n = var(prefix, D.INDEX)
        tensor = ir.TensorType([16], D.FP32, tensor_view=ir.TensorView(stride=[n], layout=ir.TensorLayout.ND))
        return function(prefix, [ir.Var(prefix + "_tensor", tensor, SPAN)])

    equal(build("n"), build("m"), False)
    equal(build("n"), build("m"), auto=True)


def test_default_hash_revisits_subtree_after_a_binding_is_introduced():
    x, y = var("x"), var("y")
    shared = add(x, x)
    one = ir.ConstInt(1, D.INT64, SPAN)
    lhs = ir.SeqStmts(
        [ir.EvalStmt(shared, SPAN), ir.AssignStmt(x, one, SPAN), ir.EvalStmt(shared, SPAN)], SPAN
    )
    rhs = ir.SeqStmts(
        [ir.EvalStmt(shared, SPAN), ir.AssignStmt(y, one, SPAN), ir.EvalStmt(add(y, y), SPAN)], SPAN
    )
    assert ir.structural_equal(lhs, rhs)
    assert ir.structural_hash(lhs) == ir.structural_hash(rhs)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
