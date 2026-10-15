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
    with pytest.raises(ir.StructuralComparisonUnsupported):
        ir.structural_equal(node, node, mode=STRICT)
    with pytest.raises(ir.StructuralComparisonUnsupported):
        ir.assert_structural_equal(node, node, mode=STRICT)


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
