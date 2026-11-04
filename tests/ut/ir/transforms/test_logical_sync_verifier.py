# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Logical synchronization is function-local SSA, including manually built IR."""

import subprocess
import sys
import textwrap

import pytest
from pypto import DataType, ir, passes
from pypto.ir.op import sync

SPAN = ir.Span("logical_sync.py", 5, 1, 5, 20)


def _token(name="ready", pipe=ir.PipeType.MTE2):
    return ir.Var(name, ir.SyncTokenType(pipe), SPAN)


def _index(value):
    return ir.ConstInt(value, DataType.INDEX, SPAN)


def _seq(*statements):
    return ir.SeqStmts(list(statements), SPAN)


def _record(token):
    return ir.AssignStmt(token, sync.record(token.type.pipe), SPAN)


def _wait(token):
    return ir.EvalStmt(sync.wait(token, ir.PipeType.V), SPAN)


def _raw(name, args=(), *, result=None, kwargs=None, attrs=None):
    # Bypass type deduction deliberately: the verifier must check stored schemas
    # and source labels rather than silently reconstructing a valid replacement.
    return ir.Call(
        ir.Op(name), list(args), kwargs or {}, attrs, result if result is not None else ir.VoidType(), SPAN
    )


def _program(body, *, kind=ir.FunctionType.AIV, params=(), returns=(), attrs=None):
    function = ir.Function("kernel", list(params), list(returns), body, SPAN, type=kind, attrs=attrs)
    return ir.Program([function], "LogicalSync", SPAN)


def _verify(program):
    properties = passes.IRPropertySet()
    properties.insert(passes.IRProperty.LogicalSyncValid)
    return passes.PropertyVerifierRegistry.verify(properties, program)


def _assert_error(program, text):
    diagnostics = _verify(program)
    assert any(text in diagnostic.message for diagnostic in diagnostics), [d.message for d in diagnostics]
    assert all(diagnostic.rule_name == "LogicalSyncValid" for diagnostic in diagnostics)


@pytest.mark.parametrize("kind", [ir.FunctionType.AIC, ir.FunctionType.AIV])
def test_specialized_device_completion_and_multiple_waits_are_valid(kind):
    ready = _token()
    assert _verify(_program(_seq(_record(ready), _wait(ready), _wait(ready)), kind=kind)) == []


@pytest.mark.parametrize(
    "kind",
    [
        ir.FunctionType.Opaque,
        ir.FunctionType.InCore,
        ir.FunctionType.Orchestration,
        ir.FunctionType.Group,
        ir.FunctionType.Spmd,
        ir.FunctionType.Inline,
        ir.FunctionType.Graph,
        ir.FunctionType.SIMT,
    ],
)
def test_logical_completion_requires_a_specialized_function(kind):
    _assert_error(_program(_record(_token()), kind=kind), "specialized AIC/AIV")


@pytest.mark.parametrize("nested", [False, True])
def test_token_cannot_cross_function_signature(nested):
    type_ = ir.SyncTokenType(ir.PipeType.MTE2)
    if nested:
        type_ = ir.TupleType([ir.TupleType([type_])])
    parameter = ir.Var("incoming", type_, SPAN)
    _assert_error(_program(_seq(), params=[parameter]), "parameter boundary")
    _assert_error(_program(_seq(), returns=[type_]), "return boundary")


def test_token_cannot_escape_through_actual_return_tuple():
    ready = _token()
    returned = ir.MakeTuple([ir.MakeTuple([ready], SPAN)], SPAN)
    _assert_error(_program(_seq(_record(ready), ir.ReturnStmt([returned], SPAN))), "function return")


def test_temporary_tuple_and_projection_preserve_local_token_ownership():
    ready = _token()
    tuple_value = ir.MakeTuple([ready, _index(1)], SPAN)
    pair = ir.Var("pair", tuple_value.type, SPAN)
    alias = _token("alias")
    body = _seq(
        _record(ready),
        ir.AssignStmt(pair, tuple_value, SPAN),
        ir.AssignStmt(alias, ir.TupleGetItemExpr(pair, 0, SPAN), SPAN),
        _wait(alias),
    )
    assert _verify(_program(body)) == []


@pytest.mark.parametrize("container", ["call", "array", "submit"])
def test_token_cannot_pass_through_opaque_calls_or_arrays(container):
    ready = _token()
    if container == "call":
        escaped = ir.Call(ir.GlobalVar("helper"), [ir.MakeTuple([ready], SPAN)], ir.VoidType(), SPAN)
    elif container == "array":
        array = ir.Var("slots", ir.ArrayType(DataType.INDEX, 4), SPAN)
        escaped = _raw("array.update_element", [array, _index(0), ready], result=array.type)
    else:
        escaped = ir.Submit(ir.GlobalVar("helper"), [ready], [], ir.ScalarType(DataType.TASK_ID), SPAN)
    _assert_error(
        _program(_seq(_record(ready), ir.EvalStmt(escaped, SPAN))), "cannot pass through ordinary calls"
    )


@pytest.mark.parametrize("nested", [False, True])
def test_completion_cannot_cross_simt_launch_arguments(nested):
    """A launch statement must enforce the same token boundary as ordinary calls."""
    ready = _token()
    argument = ir.MakeTuple([_index(1), ir.MakeTuple([ready], SPAN)], SPAN) if nested else ready
    launch = ir.SimtLaunchStmt(
        ir.GlobalVar("helper"), [_index(32), _index(1), _index(1)], [argument], [], [], SPAN
    )
    _assert_error(_program(_seq(_record(ready), launch)), "cannot pass through ordinary calls")


@pytest.mark.parametrize("kind", ["call", "submit", "simt"])
def test_logical_completion_requires_leaf_even_without_token_arguments(kind):
    ready = _token()
    if kind == "simt":
        invocation = ir.SimtLaunchStmt(
            ir.GlobalVar("helper"), [_index(32), _index(1), _index(1)], [], [], [], SPAN
        )
    else:
        call = (
            ir.Call(ir.GlobalVar("helper"), [], ir.VoidType(), SPAN)
            if kind == "call"
            else ir.Submit(ir.GlobalVar("helper"), [], [], ir.ScalarType(DataType.TASK_ID), SPAN)
        )
        invocation = ir.EvalStmt(call, SPAN)
    _assert_error(_program(_seq(_record(ready), invocation)), "leaf AIC/AIV")
    # The new property must leave preexisting programs without logical sync alone.
    assert _verify(_program(invocation)) == []


def test_simt_helper_and_scope_without_logical_sync_remain_valid():
    """Checking completion boundaries must preserve ordinary SIMT authoring and launches."""
    dims: list[ir.Expr] = [_index(32), _index(1), _index(1)]
    scope = ir.SimtScopeStmt(dims, _seq(), SPAN)
    assert _verify(_program(scope)) == []
    helper = ir.Function("helper", [], [], ir.ReturnStmt([], SPAN), SPAN, type=ir.FunctionType.SIMT)
    launch = ir.SimtLaunchStmt(ir.GlobalVar("helper"), dims, [], [], [], SPAN)
    caller = ir.Function("kernel", [], [], launch, SPAN, type=ir.FunctionType.AIV)
    assert _verify(ir.Program([helper, caller], "LogicalSync", SPAN)) == []


@pytest.mark.parametrize("location", ["function", "call", "scope"])
def test_metadata_cannot_be_a_hidden_token_carrier(location):
    ready = _token()
    if location == "function":
        program = _program(_record(ready), attrs={"hidden": ready})
    elif location == "call":
        call = _raw("system.bar_all", attrs={"hidden": ready})
        program = _program(_seq(_record(ready), ir.EvalStmt(call, SPAN)))
    else:
        scope = ir.RegScopeStmt("vf", _seq(), SPAN, attrs={"hidden": ready})
        program = _program(_seq(_record(ready), scope))
    _assert_error(program, "metadata")


@pytest.mark.parametrize("placement", ["eval", "join", "tuple", "yield"])
def test_record_requires_its_own_assignment(placement):
    record = sync.record(ir.PipeType.MTE2)
    if placement == "eval":
        body = ir.EvalStmt(record, SPAN)
    elif placement == "join":
        body = ir.AssignStmt(_token(), sync.join(record, record), SPAN)
    elif placement == "tuple":
        value = ir.MakeTuple([record], SPAN)
        body = ir.AssignStmt(ir.Var("pair", value.type, SPAN), value, SPAN)
    else:
        body = ir.IfStmt(ir.ConstBool(True, SPAN), ir.YieldStmt([record], SPAN), None, [], SPAN)
    _assert_error(_program(body), "own AssignStmt")


@pytest.mark.parametrize("name", ["sync.wait", "sync.vf_barrier"])
def test_effectful_waits_cannot_be_nested(name):
    ready = _token()
    call = _raw(
        name,
        [ready] if name == "sync.wait" else [],
        result=ir.ScalarType(DataType.INDEX),
        kwargs={"dst_pipe": int(ir.PipeType.V)} if name == "sync.wait" else {},
    )
    # Forge a value result to bypass the constructor's early rejection of Void
    # operands; schema and placement checks must still diagnose the nested wait.
    outer = _raw("system.bar_all", [call])
    _assert_error(_program(_seq(_record(ready), ir.EvalStmt(outer, SPAN))), "standalone EvalStmt")


@pytest.mark.parametrize(
    ("name", "args", "kwargs", "result", "message"),
    [
        ("sync.record", [], {}, ir.SyncTokenType(ir.PipeType.MTE2), "pipe attribute"),
        ("sync.record", [], {"pipe": int(ir.PipeType.ALL)}, ir.SyncTokenType(ir.PipeType.MTE2), "concrete"),
        ("sync.record", [], {"pipe": int(ir.PipeType.MTE2)}, ir.SyncTokenType(ir.PipeType.V), "result type"),
        ("sync.completed", [], {"pipe": int(ir.PipeType.MTE2)}, ir.ScalarType(DataType.INDEX), "result type"),
        ("sync.wait", [_index(0)], {"dst_pipe": int(ir.PipeType.V)}, ir.VoidType(), "SyncToken"),
        ("sync.join", [], {}, ir.SyncTokenType(ir.PipeType.MTE2), "exactly two"),
        ("sync.memory_overlaps", [], {}, ir.ScalarType(DataType.BOOL), "two storage handles"),
        ("sync.vf_barrier", [], {"unexpected": 1}, ir.VoidType(), "takes no operands or attributes"),
    ],
)
def test_raw_logical_call_schema_and_result_are_revalidated(name, args, kwargs, result, message):
    call = _raw(name, args, kwargs=kwargs, result=result)
    statement = (
        ir.EvalStmt(call, SPAN)
        if isinstance(result, ir.VoidType)
        else ir.AssignStmt(ir.Var("result", result, SPAN), call, SPAN)
    )
    _assert_error(_program(statement), message)


@pytest.mark.parametrize("kind", ["outside", "aic", "inside"])
def test_vf_barrier_has_aiv_region_ownership(kind):
    barrier = ir.EvalStmt(sync.vf_barrier(), SPAN)
    body = barrier if kind == "outside" else ir.RegScopeStmt("vf", barrier, SPAN)
    program = _program(body, kind=ir.FunctionType.AIC if kind == "aic" else ir.FunctionType.AIV)
    if kind == "inside":
        assert _verify(program) == []
    else:
        _assert_error(program, "RegScope in a specialized AIV")


@pytest.mark.parametrize("direction", ["capture", "create"])
def test_outer_tokens_cannot_enter_or_be_created_in_vf(direction):
    ready = _token()
    region = ir.RegScopeStmt("vf", _wait(ready) if direction == "capture" else _record(ready), SPAN)
    body = _seq(_record(ready), region) if direction == "capture" else region
    _assert_error(_program(body), "cannot enter a RegScope")


@pytest.mark.parametrize("direction", ["capture", "create"])
def test_outer_tokens_cannot_enter_or_be_created_in_simt_scope(direction):
    """An enclosing AIV annotation cannot authorize completion facts in the SIMT domain."""
    ready = _token()
    region = ir.SimtScopeStmt(
        [_index(32), _index(1), _index(1)], _wait(ready) if direction == "capture" else _record(ready), SPAN
    )
    body = _seq(_record(ready), region) if direction == "capture" else region
    _assert_error(_program(body), "requires a specialized AIC/AIV function after inlining and outlining")


def test_simt_scope_local_completion_cannot_gain_an_outer_lifetime():
    """Even invalid region-local definitions must expire when leaving the SIMT body."""
    ready = _token()
    region = ir.SimtScopeStmt([_index(32), _index(1), _index(1)], _record(ready), SPAN)
    _assert_error(_program(_seq(region, _wait(ready))), "same lexical scope")


def test_token_use_before_definition_and_multiple_definitions_are_rejected():
    ready = _token()
    _assert_error(_program(_seq(_wait(ready), _record(ready))), "defined before use")
    _assert_error(_program(_seq(_record(ready), _record(ready))), "multiple SSA definitions")


def test_branch_local_token_cannot_leak_without_explicit_result():
    ready = _token()
    branch = ir.IfStmt(ir.ConstBool(True, SPAN), _record(ready), None, [], SPAN)
    _assert_error(_program(_seq(branch, _wait(ready))), "same lexical scope")


def test_explicit_branch_result_is_valid_and_preserves_source_type():
    first, second, result = _token("first"), _token("second"), _token("result")
    branch = ir.IfStmt(
        ir.ConstBool(True, SPAN),
        _seq(_record(first), ir.YieldStmt([first], SPAN)),
        _seq(ir.AssignStmt(second, sync.completed(ir.PipeType.MTE2), SPAN), ir.YieldStmt([second], SPAN)),
        [result],
        SPAN,
    )
    assert _verify(_program(_seq(branch, _wait(result)))) == []
    forged = _token("forged", ir.PipeType.V)
    _assert_error(
        _program(_seq(branch, ir.AssignStmt(forged, result, SPAN))), "source-pipeline/type mismatch"
    )


@pytest.mark.parametrize("missing", ["else", "yield", "count", "source"])
def test_token_branch_result_requires_matching_terminal_yields(missing):
    ready, result = _token(), _token("result")
    then = _seq(_record(ready), ir.YieldStmt([ready], SPAN))
    other = _token("other", ir.PipeType.V if missing == "source" else ir.PipeType.MTE2)
    otherwise = _seq(_record(other), ir.YieldStmt([] if missing == "count" else [other], SPAN))
    if missing == "yield":
        otherwise = _record(other)
    branch = ir.IfStmt(
        ir.ConstBool(True, SPAN), then, None if missing == "else" else otherwise, [result], SPAN
    )
    _assert_error(
        _program(branch),
        "mismatch" if missing in ("count", "source") else "yield" if missing == "else" else "YieldStmt",
    )


@pytest.mark.parametrize("kind", ["for", "while"])
def test_loop_token_carry_has_explicit_lexical_lifetime(kind):
    initial, result = _token("initial"), _token("result")
    carried = ir.IterArg("carried", initial.type, initial, SPAN)
    fresh = _token("fresh")
    body = _seq(_wait(carried), _record(fresh), ir.YieldStmt([fresh], SPAN))
    loop = (
        ir.ForStmt(
            ir.Var("i", ir.ScalarType(DataType.INDEX), SPAN),
            _index(0),
            _index(4),
            _index(1),
            [carried],
            body,
            [result],
            SPAN,
        )
        if kind == "for"
        else ir.WhileStmt(ir.ConstBool(False, SPAN), [carried], body, [result], SPAN)
    )
    assert _verify(_program(_seq(_record(initial), loop, _wait(result)))) == []
    _assert_error(_program(_seq(_record(initial), loop, _wait(fresh))), "same lexical scope")
    _assert_error(_program(_seq(_record(initial), loop, _wait(carried))), "same lexical scope")


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("used", [False, True])
def test_for_induction_variable_cannot_manufacture_completion(nested, used):
    """Raw IR must not turn a numeric induction binding into a completion fact."""
    token_type = ir.SyncTokenType(ir.PipeType.MTE2)
    induction_type = (
        ir.TupleType([ir.ScalarType(DataType.INDEX), ir.TupleType([token_type])]) if nested else token_type
    )
    induction = ir.Var("i", induction_type, SPAN)
    token = ir.TupleGetItemExpr(ir.TupleGetItemExpr(induction, 1, SPAN), 0, SPAN) if nested else induction
    # No record, completed value or incoming carry provides this token. Reject
    # the induction type even when unused, before any wait can legitimize it.
    loop = ir.ForStmt(
        induction, _index(0), _index(4), _index(1), [], _wait(token) if used else _seq(), [], SPAN
    )
    _assert_error(_program(loop), "induction variable cannot contain SyncToken")


def test_loop_declared_source_cannot_relabel_its_initial_token():
    initial = _token("initial")
    carried = ir.IterArg("carried", ir.SyncTokenType(ir.PipeType.V), initial, SPAN)
    result = _token("result", ir.PipeType.V)
    loop = ir.WhileStmt(ir.ConstBool(False, SPAN), [carried], ir.YieldStmt([carried], SPAN), [result], SPAN)
    _assert_error(_program(_seq(_record(initial), loop)), "source-pipeline/type mismatch")


def test_logical_sync_property_is_verified_at_every_structural_boundary():
    for properties in (
        passes.get_verified_properties(),
        passes.get_structural_properties(),
        passes.get_default_verify_properties(),
    ):
        assert properties.contains(passes.IRProperty.LogicalSyncValid)


@pytest.mark.parametrize("kind", ["arithmetic", "cast", "condition"])
def test_token_cannot_be_reinterpreted_as_scalar_data(kind):
    ready = _token()
    if kind == "arithmetic":
        statement = ir.EvalStmt(ir.Eq(ready, ready, DataType.BOOL, SPAN), SPAN)
        message = "scalar arithmetic or comparison"
    elif kind == "cast":
        statement = ir.EvalStmt(ir.Cast(ready, DataType.INDEX, SPAN), SPAN)
        message = "cast"
    else:
        statement = ir.IfStmt(ready, _seq(), None, [], SPAN)
        message = "condition"
    _assert_error(_program(_seq(_record(ready), statement)), message)


def test_temporary_nested_tuple_can_cross_an_explicit_branch_result():
    ready = _token()
    value = ir.MakeTuple([ir.MakeTuple([ready], SPAN), _index(1)], SPAN)
    result = ir.Var("result", value.type, SPAN)
    branch = ir.IfStmt(
        ir.ConstBool(True, SPAN),
        ir.YieldStmt([value], SPAN),
        ir.YieldStmt([value], SPAN),
        [result],
        SPAN,
    )
    inner = ir.TupleGetItemExpr(result, 0, SPAN)
    projected = ir.TupleGetItemExpr(inner, 0, SPAN)
    assert _verify(_program(_seq(_record(ready), branch, _wait(projected)))) == []


@pytest.mark.parametrize("fault", ["destination", "join_source", "overlap_handle", "overlap_bound"])
def test_raw_operand_contracts_cannot_bypass_the_registry(fault):
    first, second = _token("first"), _token("second", ir.PipeType.V)
    prefix = [_record(first), _record(second)]
    if fault == "destination":
        call = _raw("sync.wait", [first], kwargs={"dst_pipe": int(ir.PipeType.ALL)})
        message = "concrete"
    elif fault == "join_source":
        call = _raw("sync.join", [first, second], result=first.type)
        message = "same source pipe"
    else:
        handle = ir.Var("storage", ir.TensorType([_index(64)], DataType.FP32), SPAN)
        bounds = [_index(0), _index(64), _index(0), _index(64)]
        arguments = [handle, handle, *bounds]
        if fault == "overlap_handle":
            arguments[0] = _index(0)
            message = "storage handles"
        else:
            arguments[2] = ir.ConstInt(0, DataType.INT64, SPAN)
            message = "ScalarType(INDEX)"
        call = _raw("sync.memory_overlaps", arguments, result=ir.ScalarType(DataType.BOOL))
    statement = (
        ir.EvalStmt(call, SPAN)
        if isinstance(call.type, ir.VoidType)
        else ir.AssignStmt(ir.Var("result", call.type, SPAN), call, SPAN)
    )
    _assert_error(_program(_seq(*prefix, statement)), message)


def test_logical_operation_cannot_be_encoded_as_task_launch():
    """Public raw Submit construction cannot bypass logical operation ownership."""
    submit = ir.Submit(ir.get_op("sync.vf_barrier"), [], [], ir.ScalarType(DataType.TASK_ID), SPAN)
    _assert_error(_program(ir.EvalStmt(submit, SPAN)), "require Call")


@pytest.mark.parametrize("location", ["parameter", "assignment", "tuple"])
def test_completion_cannot_hide_in_type_geometry(location):
    ready = _token()
    # ShapedType construction rejects Void dimensions but permits other Expr
    # values. A completion used as geometry must still be rejected at the IR
    # boundary even when the surrounding Tensor is not a token-containing type.
    malformed = ir.TensorType([ready], DataType.FP32)
    value = ir.Var("value", malformed, SPAN)
    if location == "parameter":
        program = _program(_record(ready), params=[value])
    elif location == "assignment":
        alias = ir.Var("alias", malformed, SPAN)
        program = _program(_seq(_record(ready), ir.AssignStmt(alias, value, SPAN)))
    else:
        packed = ir.MakeTuple([value], SPAN)
        program = _program(_seq(_record(ready), ir.EvalStmt(packed, SPAN)))
    _assert_error(program, "metadata")


@pytest.mark.parametrize("field", ["base", "byte_offset"])
@pytest.mark.parametrize("owner", ["tensor", "tile", "direct"])
def test_completion_cannot_hide_in_memref_metadata(field, owner):
    ready = _token()
    base = ready if field == "base" else ir.Var("base", ir.PtrType(), SPAN)
    offset = ready if field == "byte_offset" else _index(0)
    # MemRef construction checks for Void operands, not completion facts. The
    # generic type-expression walker exposes slot_index only, so the logical
    # verifier must additionally reject tokens hidden in base/resolved offsets.
    memref = ir.MemRef(base, offset, 256, SPAN)
    if owner == "tensor":
        type_ = ir.TensorType([_index(64)], DataType.FP32, memref)
        value = ir.Var("value", type_, SPAN)
    elif owner == "tile":
        type_ = ir.TileType([_index(64)], DataType.FP32, memref, memory_space=ir.MemorySpace.Vec)
        value = ir.Var("value", type_, SPAN)
    else:
        value = memref
    _assert_error(_program(_seq(_record(ready), ir.EvalStmt(value, SPAN))), "metadata")


@pytest.mark.parametrize("field", ["base", "size"])
@pytest.mark.parametrize("owner", ["direct", "distributed_tensor"])
def test_completion_cannot_hide_in_window_metadata(field, owner):
    """Window address/size fields are metadata even when the token is visible."""
    ready = _token()
    base = ready if field == "base" else ir.Var("base", ir.PtrType(), SPAN)
    size = ready if field == "size" else _index(256)
    window = ir.WindowBuffer(base, size, span=SPAN)
    value = (
        window
        if owner == "direct"
        else ir.Var("view", ir.DistributedTensorType([_index(64)], DataType.FP32, window), SPAN)
    )
    _assert_error(_program(_seq(_record(ready), ir.EvalStmt(value, SPAN))), "metadata")


@pytest.mark.parametrize("field", ["base", "size"])
@pytest.mark.parametrize("signature", ["window_parameter", "tensor_parameter", "tensor_return"])
def test_signature_window_metadata_rejects_orphan_token_without_sync_calls(field, signature):
    """Signature validation must not depend on first finding a sync operation."""
    orphan = _token("orphan")
    base = orphan if field == "base" else ir.Var("base", ir.PtrType(), SPAN)
    size = orphan if field == "size" else _index(256)
    window = ir.WindowBuffer(base, size, span=SPAN)
    tensor_type = ir.DistributedTensorType([_index(64)], DataType.FP32, window)
    if signature == "window_parameter":
        program = _program(_seq(), params=[window])
    elif signature == "tensor_parameter":
        program = _program(_seq(), params=[ir.Var("view", tensor_type, SPAN)])
    else:
        program = _program(_seq(), returns=[ir.TupleType([tensor_type])])
    _assert_error(program, "metadata")


@pytest.mark.parametrize("direct", [False, True])
def test_ordinary_window_pointer_and_scalar_metadata_remain_valid(direct):
    """Checking hidden completion values must preserve normal window metadata."""
    size = ir.Var("bytes", ir.ScalarType(DataType.INDEX), SPAN)
    window = ir.WindowBuffer(ir.Var("base", ir.PtrType(), SPAN), size, span=SPAN)
    tensor = ir.Var("view", ir.DistributedTensorType([_index(64)], DataType.FP32, window), SPAN)
    value = window if direct else tensor
    assert _verify(_program(ir.EvalStmt(value, SPAN), params=[size, tensor])) == []


def _shared_completion_expression(ready, form):
    """Build one expression identity reused across different lexical contexts."""
    if form == "join":
        inner = sync.join(ready, ready)
        return sync.join(inner, inner)
    inner = ir.MakeTuple([ready, _index(0)], SPAN)
    return ir.MakeTuple([inner, inner], SPAN)


def _use_shared_completion(expression, form):
    """Use a pure token expression without introducing another SSA definition."""
    return _wait(expression) if form == "join" else ir.EvalStmt(expression, SPAN)


@pytest.mark.parametrize("form", ["join", "tuple"])
def test_shared_completion_expression_respects_definition_order(form):
    ready = _token()
    shared = _shared_completion_expression(ready, form)
    use = _use_shared_completion(shared, form)
    _assert_error(_program(_seq(use, _record(ready), use)), "defined before use")
    assert _verify(_program(_seq(_record(ready), use, use))) == []


@pytest.mark.parametrize("form", ["join", "tuple"])
@pytest.mark.parametrize("escape", ["sibling_branch", "after_loop", "metadata", "vf", "simt"])
def test_shared_completion_expression_rechecks_each_use_context(form, escape):
    """A cached legal use cannot authorize the same DAG in a different scope."""
    ready = _token()
    shared = _shared_completion_expression(ready, form)
    use = _use_shared_completion(shared, form)
    if escape == "sibling_branch":
        body = ir.IfStmt(ir.ConstBool(True, SPAN), _seq(_record(ready), use), use, [], SPAN)
        message = "same lexical scope"
    elif escape == "after_loop":
        loop = ir.ForStmt(
            ir.Var("i", ir.ScalarType(DataType.INDEX), SPAN),
            _index(0),
            _index(4),
            _index(1),
            [],
            _seq(_record(ready), use),
            [],
            SPAN,
        )
        body = _seq(loop, use)
        message = "same lexical scope"
    elif escape == "metadata":
        metadata_use = ir.EvalStmt(_raw("system.bar_all", attrs={"hidden": shared}), SPAN)
        body = _seq(_record(ready), use, metadata_use)
        message = "metadata"
    elif escape == "simt":
        body = _seq(_record(ready), use, ir.SimtScopeStmt([_index(32), _index(1), _index(1)], use, SPAN))
        message = "requires a specialized AIC/AIV function after inlining and outlining"
    else:
        body = _seq(_record(ready), use, ir.RegScopeStmt("vf", use, SPAN))
        message = "cannot enter a RegScope"
    _assert_error(_program(body), message)


@pytest.mark.parametrize("form", ["join", "tuple"])
def test_shared_completed_fact_is_rechecked_at_vf_boundary(form):
    """A pure completed fact also belongs to the outer execution domain."""
    shared = _shared_completion_expression(sync.completed(ir.PipeType.MTE2), form)
    use = _use_shared_completion(shared, form)
    _assert_error(_program(_seq(use, ir.RegScopeStmt("vf", use, SPAN))), "cannot enter a RegScope")


@pytest.mark.parametrize("nested_first", [False, True])
def test_shared_record_expression_rechecks_its_capture_position(nested_first):
    """A legal assignment must not authorize the same record at a nested site."""
    capture = sync.record(ir.PipeType.MTE2)
    ready, combined = _token("ready"), _token("combined")
    legal = ir.AssignStmt(ready, capture, SPAN)
    nested = ir.AssignStmt(combined, sync.join(capture, capture), SPAN)
    statements = (nested, legal) if nested_first else (legal, nested)
    _assert_error(_program(_seq(*statements, _wait(ready))), "own AssignStmt")


@pytest.mark.parametrize("form", ["join", "tuple"])
def test_shared_completion_dag_verification_does_not_expand_paths_exponentially(form):
    """A 60-level DAG has O(60) nodes but more than 2**60 unfolded paths.

    Isolate the regression in a child interpreter: an accidental recursive
    expansion times out this child instead of hanging the entire pytest worker.
    The timeout is a generous liveness bound, not a machine-speed ratio.
    """
    script = textwrap.dedent("""\
        import sys
        from pypto import ir, passes
        from pypto.ir.op import sync

        span = ir.Span.unknown()
        shared = sync.completed(ir.PipeType.MTE2)
        for _ in range(60):
            if sys.argv[1] == "join":
                shared = sync.join(shared, shared)
            else:
                shared = ir.MakeTuple([shared, shared], span)
        expression = sync.wait(shared, ir.PipeType.V) if sys.argv[1] == "join" else shared
        function = ir.Function("kernel", [], [], ir.EvalStmt(expression, span), span,
                               type=ir.FunctionType.AIV)
        program = ir.Program([function], "SharedCompletionDag", span)
        properties = passes.IRPropertySet()
        properties.insert(passes.IRProperty.LogicalSyncValid)
        diagnostics = passes.PropertyVerifierRegistry.verify(properties, program)
        assert diagnostics == [], [diagnostic.message for diagnostic in diagnostics]
        """)
    result = subprocess.run(
        [sys.executable, "-c", script, form], capture_output=True, text=True, check=False, timeout=20
    )
    assert result.returncode == 0, result.stdout + result.stderr


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
