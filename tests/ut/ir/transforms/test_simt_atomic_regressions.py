# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Regression coverage for atomic coordinates and observable nested calls."""

import pytest
from pypto import DataType, Error, backend, codegen, ir, passes
from pypto.ir.op import simt
from pypto.ir.op.distributed import get_comm_ctx, rank

SPAN = ir.Span.unknown()


@pytest.fixture
def a5_backend():
    backend.reset_for_testing()
    backend.set_backend_type(backend.BackendType.Ascend950)
    yield
    backend.reset_for_testing()


def collect_calls(program, name):
    calls = []

    class Collector(ir.IRVisitor):
        def visit_call(self, op):
            if op.op.name == ir.get_op(name).name:
                calls.append(op)
            super().visit_call(op)

    Collector().visit_program(program)
    return calls


@pytest.mark.parametrize("name", ["atomic_add", "atomic_min", "atomic_max", "atomic_exch", "atomic_cas"])
@pytest.mark.parametrize("storage_type", [ir.TileType, ir.TensorType])
def test_atomic_flattening_preserves_logical_element(name, storage_type):
    storage = ir.Var("storage", storage_type([2, 3, 32], DataType.INT32), SPAN)
    value = ir.ConstInt(1, DataType.INT32, SPAN)
    args = [storage, [1, 2, 7], value] + ([value] if name == "atomic_cas" else [])
    call = getattr(simt, name)(*args)
    function = ir.Function(
        "helper",
        [(storage, ir.ParamDirection.InOut)],
        [],
        ir.EvalStmt(call, SPAN),
        SPAN,
        ir.FunctionType.SIMT,
    )
    program = ir.Program([function], "Atomic", SPAN)
    result = passes.simplify()(passes.flatten_tile_nd_to_2d()(program))
    rewritten = collect_calls(result, f"simt.{name}")[0]
    indices = [i.value for i in rewritten.args[1].elements]
    if storage_type is ir.TileType:
        assert indices == [5, 7]
        assert [d.value for d in rewritten.args[0].type.shape] == [6, 32]
    else:
        assert indices == [1, 2, 7]
        assert [d.value for d in rewritten.args[0].type.shape] == [2, 3, 32]


@pytest.mark.parametrize("expression", ["subtract", "multiply_zero", "equal", "tuple"])
def test_simplify_preserves_nested_atomic_occurrences(expression):
    storage = ir.Var("storage", ir.TensorType([1], DataType.INT32), SPAN)
    one = ir.ConstInt(1, DataType.INT32, SPAN)
    first, second = (simt.atomic_add(storage, [0], one) for _ in range(2))
    if expression == "subtract":
        value = ir.Sub(first, second, DataType.INT32, SPAN)
        count = 2
    elif expression == "multiply_zero":
        value = ir.Mul(first, ir.ConstInt(0, DataType.INT32, SPAN), DataType.INT32, SPAN)
        count = 1
    elif expression == "equal":
        value = ir.Eq(first, second, DataType.BOOL, SPAN)
        count = 2
    else:
        value = ir.TupleGetItemExpr(
            ir.MakeTuple([ir.Sub(first, second, DataType.INT32, SPAN)], SPAN), 0, SPAN
        )
        count = 2
    result = ir.Var("result", value.type, SPAN)
    function = ir.Function(
        "helper",
        [(storage, ir.ParamDirection.InOut)],
        [],
        ir.AssignStmt(result, value, SPAN),
        SPAN,
        ir.FunctionType.SIMT,
    )
    program = ir.Program([function], "Atomic", SPAN)
    simplified = passes.simplify()(program)
    assert len(collect_calls(simplified, "simt.atomic_add")) == count


@pytest.mark.parametrize("effect_position", [0, 1])
@pytest.mark.parametrize("via_alias", [False, True])
@pytest.mark.parametrize("call_kind", ["call", "submit"])
def test_composite_lowering_preserves_tuple_effects(effect_position, via_alias, call_kind):
    # Calls and submits both lack a purity contract. Even an unselected tuple
    # slot must be evaluated once; alias propagation must not inline it again.
    one = ir.ConstInt(1, DataType.INT32, SPAN)
    effect = (
        ir.Call(ir.GlobalVar("effect"), [], ir.ScalarType(DataType.INT32), SPAN)
        if call_kind == "call"
        else ir.Submit(ir.GlobalVar("effect"), [], [], ir.TupleType([ir.ScalarType(DataType.TASK_ID)]), SPAN)
    )
    elements: list[ir.Expr] = [one, one]
    elements[effect_position] = effect
    pair = ir.MakeTuple(elements, SPAN)
    statements = []
    value: ir.Expr = pair
    if via_alias:
        alias = ir.Var("alias", pair.type, SPAN)
        statements.append(ir.AssignStmt(alias, pair, SPAN))
        value = alias
    projected = ir.TupleGetItemExpr(value, 1 - effect_position, SPAN)
    result = ir.Var("result", projected.type, SPAN)
    statements.append(ir.AssignStmt(result, projected, SPAN))
    body = ir.SeqStmts(statements, SPAN) if len(statements) > 1 else statements[0]
    function = ir.Function("helper", [], [], body, SPAN)
    program = ir.Program([function], "TupleEffects", SPAN)
    # This isolated pre-normalization IR fragment includes a nested Submit,
    # which the public DSL requires binding directly. Keep property verification
    # and structural comparison without round-tripping through that syntax.
    with passes.PassContext([passes.VerificationInstrument(passes.VerificationMode.BEFORE_AND_AFTER)]):
        lowered = passes.lower_composite_ops()(program)
    ir.assert_structural_equal(lowered, program)


def test_composite_lowering_retains_while_condition_tuple_atomic():
    storage = ir.Var("storage", ir.TensorType([1], DataType.INT32), SPAN)
    one = ir.ConstInt(1, DataType.INT32, SPAN)
    pair = ir.MakeTuple([simt.atomic_add(storage, [0], one), ir.ConstBool(False, SPAN)], SPAN)
    loop = ir.WhileStmt(ir.TupleGetItemExpr(pair, 1, SPAN), [], ir.SeqStmts([], SPAN), [], SPAN)
    function = ir.Function(
        "helper", [(storage, ir.ParamDirection.InOut)], [], loop, SPAN, ir.FunctionType.SIMT
    )
    program = ir.Program([function], "TupleWhile", SPAN)
    ir.assert_structural_equal(passes.lower_composite_ops()(program), program)


def test_simplify_does_not_reuse_effectful_branch_constraints():
    storage = ir.Var("storage", ir.TensorType([1], DataType.INT32), SPAN)
    one = ir.ConstInt(1, DataType.INT32, SPAN)
    zero = ir.ConstInt(0, DataType.INT32, SPAN)

    def predicate():
        return ir.Eq(simt.atomic_add(storage, [0], one), zero, DataType.BOOL, SPAN)

    nested = ir.IfStmt(predicate(), ir.EvalStmt(simt.threadfence(), SPAN), None, [], SPAN)
    body = ir.IfStmt(predicate(), nested, None, [], SPAN)
    function = ir.Function(
        "helper", [(storage, ir.ParamDirection.InOut)], [], body, SPAN, ir.FunctionType.SIMT
    )
    # The first call may return zero; the next call then returns one. A true
    # outer condition is not a constraint on a new evaluation of the same op.
    simplified = passes.simplify()(ir.Program([function], "Atomic", SPAN))
    assert len(collect_calls(simplified, "simt.atomic_add")) == 2


@pytest.mark.parametrize("name", ["atomic_add", "atomic_min", "atomic_max", "atomic_exch", "atomic_cas"])
@pytest.mark.parametrize("direction", [ir.ParamDirection.In, ir.ParamDirection.Out, ir.ParamDirection.InOut])
def test_window_atomic_outlining_permissions_and_codegen(a5_backend, name, direction):
    # A window is local GM. Its padded row stride must survive outlining, and
    # it must carry the same permissions/effects as an ordinary Tensor.
    window_type = ir.DistributedTensorType(
        [ir.ConstInt(d, DataType.INDEX, SPAN) for d in (2, 3)],
        DataType.INT32,
        tensor_view=ir.TensorView(stride=[8, 1], layout=ir.TensorLayout.ND),
    )
    window = ir.Var("window", window_type, SPAN)
    one = ir.ConstInt(1, DataType.INT32, SPAN)
    args = [window, [1, 2], one] + ([one] if name == "atomic_cas" else [])
    region = ir.SimtScopeStmt([one, one, one], ir.EvalStmt(getattr(simt, name)(*args), SPAN), SPAN)
    query = get_comm_ctx(window)
    ctx = ir.Var("ctx", query.type, SPAN)
    rank_value = rank(ctx)
    rank_var = ir.Var("rank", rank_value.type, SPAN)
    kernel = ir.Function(
        "kernel",
        [(window, direction)],
        [window_type],
        ir.SeqStmts(
            [
                region,
                ir.AssignStmt(ctx, query, SPAN),
                ir.AssignStmt(rank_var, rank_value, SPAN),
                ir.ReturnStmt([window], SPAN),
            ],
            SPAN,
        ),
        SPAN,
        ir.FunctionType.AIV,
    )
    program = ir.Program([kernel], "WindowAtomic", SPAN)
    if direction != ir.ParamDirection.InOut:
        with pytest.raises((Error, ValueError), match="enclosing In parameter|initialized storage"):
            passes.outline_simt_scopes()(program)
        return

    outlined = passes.convert_to_ssa()(passes.outline_simt_scopes()(program))
    selected = passes.IRPropertySet()
    selected.insert(passes.IRProperty.SimtValid)
    assert not passes.PropertyVerifierRegistry.verify(selected, outlined)
    ir.verify_structural_bindings(outlined)
    # The enclosing device kernel needs the runtime communication context;
    # the SIMT helper's borrowed local pointer ABI must not grow a ctx argument.
    outlined = passes.materialize_dist_tensor_ctx()(passes.normalize_return_order()(outlined))
    assert not passes.PropertyVerifierRegistry.verify(selected, outlined)
    assert not collect_calls(outlined, "pld.system.get_comm_ctx")
    assert len(collect_calls(outlined, "pld.system.rank")) == 1
    bundle = codegen.PTOCodegen().generate_bundle(passes.specialize_simt_helpers()(outlined))
    helper = bundle.helpers[0]
    assert helper.parameter_types == ["!pto.ptr<i32, gm>"]
    assert helper.parameter_effects == ["readwrite"]
    assert 'pto.tileop.effects = ["readwrite"]' in bundle.caller_source
    assert f"pto.{name} " in helper.source
    assert "arith.constant 8 : index" in helper.source


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
