# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Explicit device-buffer representation and original registered-call contracts."""

import pytest
from pypto import DataType, ir, passes

SPAN = ir.Span.unknown()


def _buffer_type(rows=16):
    return ir.BufferType([rows, 32], DataType.FP32, ir.MemorySpace.Vec)


def _var(name, type_=None):
    return ir.Var(name, type_ if type_ is not None else _buffer_type(), SPAN)


def _int(value):
    return ir.ConstInt(value, DataType.INDEX, SPAN)


def _call(name, args, type_=None, kwargs=None, attrs=None):
    # Construct directly to test stored types and malformed calls that bypass
    # registry creation. The verifier must not repair these by recreating them.
    return ir.Call(
        ir.Op(name), args, kwargs or {}, attrs, type_ if type_ is not None else ir.VoidType(), SPAN
    )


def _program(body=None, params=(), returns=(), func_type=ir.FunctionType.InCore, attrs=None):
    function = ir.Function(
        "kernel",
        list(params),
        list(returns),
        body if body is not None else ir.ReturnStmt(SPAN),
        SPAN,
        type=func_type,
        attrs=attrs,
    )
    return ir.Program([function], "Buffers", SPAN)


def _allocation(type_):
    return _call("buffer.alloc", [ir.MakeTuple([], SPAN)], type_)


def _verify(program):
    props = passes.IRPropertySet()
    props.insert(passes.IRProperty.BufferIR)
    return passes.PropertyVerifierRegistry.verify(props, program)


def _assert_error(program, message, rule="BufferIR"):
    diagnostics = _verify(program)
    assert any(d.rule_name == rule and message in d.message for d in diagnostics), [
        (d.rule_name, d.message) for d in diagnostics
    ]


@pytest.mark.parametrize("func_type", [ir.FunctionType.InCore, ir.FunctionType.AIC, ir.FunctionType.AIV])
def test_incoming_buffers_and_explicit_writes_are_valid(func_type):
    source, destination = _var("source"), _var("destination")
    call = _call("buffer.copy", [source, destination])
    assert _verify(_program(ir.EvalStmt(call, SPAN), [source, destination], func_type=func_type)) == []


def test_explicit_allocation_defines_a_buffer():
    destination = _var("destination")
    allocation = _allocation(destination.type)
    assert _verify(_program(ir.AssignStmt(destination, allocation, SPAN))) == []


@pytest.mark.parametrize("nested", [False, True])
def test_incoming_buffer_parameter_cannot_be_redefined(nested):
    incoming = _var("incoming", ir.TupleType([_buffer_type()]) if nested else _buffer_type())
    params = [incoming]
    if nested:
        outer = _var("outer", ir.TupleType([incoming.type]))
        params.append(outer)
        value = ir.TupleGetItemExpr(outer, 0, SPAN)
    else:
        value = _allocation(incoming.type)
    _assert_error(
        _program(ir.AssignStmt(incoming, value, SPAN), params),
        "Incoming buffer parameter 'incoming' cannot be redefined",
    )


def test_allocation_cannot_be_discarded_as_eval():
    _assert_error(_program(ir.EvalStmt(_allocation(_buffer_type()), SPAN)), "require a direct AssignStmt")


def test_nested_allocation_must_first_define_an_ssa_handle():
    destination = _var("destination")
    call = _call("buffer.copy", [_allocation(destination.type), destination])
    _assert_error(_program(ir.EvalStmt(call, SPAN), [destination]), "require a direct AssignStmt")


def test_nested_buffer_write_must_be_a_direct_eval():
    source = _var("source")
    write = _call("buffer.copy", [source, source])
    # Void cannot be a positional operand. An Expr-valued compiler attribute
    # still exercises the verifier's exact direct-statement Call identity.
    wrapper = _call("scalar_helper", [], ir.ScalarType(DataType.INDEX), attrs={"hidden_write": write})
    _assert_error(_program(ir.EvalStmt(wrapper, SPAN), [source]), "require a direct EvalStmt")


def test_nested_tuple_parameter_projection_is_valid():
    incoming = _var("incoming", ir.TupleType([ir.TupleType([_buffer_type()]), ir.ScalarType(DataType.INDEX)]))
    inner = _var("inner", ir.TupleType([_buffer_type()]))
    buffer = _var("buffer")
    body = ir.SeqStmts(
        [
            ir.AssignStmt(inner, ir.TupleGetItemExpr(incoming, 0, SPAN), SPAN),
            ir.AssignStmt(buffer, ir.TupleGetItemExpr(inner, 0, SPAN), SPAN),
            ir.EvalStmt(_call("buffer.mul", [buffer, buffer, buffer]), SPAN),
        ],
        SPAN,
    )
    assert _verify(_program(body, [incoming])) == []


def test_gm_tensor_memref_metadata_remains_valid():
    memref = ir.MemRef(_var("gm_base", ir.PtrType()), 0, 2048, SPAN)
    tensor = _var("tensor", ir.TensorType([16, 32], DataType.FP32, memref))
    assert _verify(_program(ir.ReturnStmt([tensor], SPAN), [tensor], [tensor.type])) == []


@pytest.mark.parametrize("tensor_type", [ir.TensorType, ir.DistributedTensorType])
@pytest.mark.parametrize("field", ["shape", "valid_shape", "stride"])
def test_tensor_type_metadata_cannot_hide_buffer_references(tensor_type, field):
    hidden = _var("hidden")
    shape = [hidden] if field == "shape" else [_int(16)]
    view = ir.TensorView(
        [hidden] if field == "stride" else [],
        ir.TensorLayout.ND,
        [hidden] if field == "valid_shape" else [],
    )
    tensor = _var("gm", tensor_type(shape, DataType.FP32, None, view))
    # Shared SSA admits scalar type-dynamic signature variables. A storage
    # handle in the same position must not become an implicit definition.
    _assert_error(_program(params=[tensor]), "Type metadata cannot carry buffer handles")


def test_nested_return_type_metadata_cannot_hide_wrapped_buffer_references():
    hidden = _var("hidden")
    wrapped = ir.Add(hidden, _int(1), DataType.INDEX, SPAN)
    view = ir.TensorView([], ir.TensorLayout.ND, [wrapped])
    tensor_type = ir.TensorType([16], DataType.FP32, None, view)
    return_type = ir.TupleType([ir.TupleType([tensor_type, tensor_type])])
    _assert_error(_program(returns=[return_type]), "Type metadata cannot carry buffer handles")


@pytest.mark.parametrize("tensor_type", [ir.TensorType, ir.DistributedTensorType])
@pytest.mark.parametrize("field", ["base", "byte_offset", "wrapped_byte_offset"])
def test_gm_memref_metadata_cannot_hide_buffer_references(tensor_type, field):
    hidden = _var("hidden")
    base = hidden if field == "base" else _var("gm_base", ir.PtrType())
    offset = _int(0) if field == "base" else hidden
    if field == "wrapped_byte_offset":
        offset = ir.Add(hidden, _int(4), DataType.INDEX, SPAN)
    memref = ir.MemRef(base, offset, 2048, SPAN)
    first = _var("first", tensor_type([16], DataType.FP32, memref))
    second = _var("second", tensor_type([32], DataType.FP32, memref))
    # Different tensor descriptors may share the same allocation carrier.
    _assert_error(_program(params=[first, second]), "Type metadata cannot carry buffer handles")


def test_gm_pointer_carrier_exemption_does_not_hide_child_buffer_reference():
    hidden = _var("hidden")
    base = ir.IterArg("gm_base", ir.PtrType(), hidden, SPAN)
    memref = ir.MemRef(base, 0, 2048, SPAN)
    tensor = _var("gm", ir.TensorType([16], DataType.FP32, memref))
    _assert_error(_program(params=[tensor]), "Type metadata cannot carry buffer handles")


@pytest.mark.parametrize("tensor_type", [ir.TensorType, ir.DistributedTensorType])
def test_scalar_gm_type_metadata_remains_valid(tensor_type):
    extent = _var("extent", ir.ScalarType(DataType.INDEX))
    stride = _var("stride", ir.ScalarType(DataType.INDEX))
    offset = ir.Add(extent, _int(4), DataType.INDEX, SPAN)
    memref = ir.MemRef(_var("gm_base", ir.PtrType()), offset, 2048, SPAN)
    view = ir.TensorView([stride], ir.TensorLayout.ND, [ir.Add(extent, _int(1), DataType.INDEX, SPAN)])
    tensor = _var("gm", tensor_type([extent], DataType.FP32, memref, view))
    assert _verify(_program(ir.ReturnStmt([tensor], SPAN), [tensor], [tensor.type])) == []


def test_orchestration_is_outside_the_property():
    tile = _var("tile", ir.TileType([16, 32], DataType.FP32))
    # Even an undefined logical tile in orchestration is outside this property;
    # composed structural verifiers must run on device functions only.
    assert _verify(_program(ir.ReturnStmt([tile], SPAN), func_type=ir.FunctionType.Orchestration)) == []


@pytest.mark.parametrize("nested", [False, True])
def test_logical_tile_parameters_are_rejected(nested):
    type_ = ir.TileType([16, 32], DataType.FP32)
    if nested:
        type_ = ir.TupleType([ir.ScalarType(DataType.INDEX), ir.TupleType([type_])])
    _assert_error(_program(params=[_var("tile", type_)]), "Logical TileType")


def test_logical_tile_return_type_is_rejected():
    _assert_error(
        _program(returns=[ir.TupleType([ir.TileType([16, 32], DataType.FP32)])]), "Logical TileType"
    )


@pytest.mark.parametrize(
    "type_", [ir.PtrType(), ir.MemRef("raw_type", SPAN).type, ir.TupleType([ir.PtrType()])]
)
def test_raw_storage_parameters_are_rejected(type_):
    _assert_error(_program(params=[_var("raw", type_)]), "Standalone MemRef/Ptr")


def test_standalone_memref_expression_is_rejected():
    memref = ir.MemRef("raw", SPAN)
    _assert_error(_program(ir.EvalStmt(memref, SPAN)), "Standalone MemRef/Ptr")


@pytest.mark.parametrize("name", ["tile.mul", "tile.unregistered", "pld.tile.unregistered"])
def test_logical_tile_family_is_rejected_even_without_tile_results(name):
    call = _call(name, [], ir.ScalarType(DataType.INDEX))
    _assert_error(_program(ir.EvalStmt(call, SPAN)), "Logical tile operation")


@pytest.mark.parametrize("name", ["buffer.unregistered", "user_function", "system.unknown"])
def test_unregistered_calls_cannot_consume_buffers(name):
    buffer = _var("buffer")
    _assert_error(_program(ir.EvalStmt(_call(name, [buffer]), SPAN), [buffer]), "cannot carry buffer handles")


def test_unregistered_buffer_operation_requires_contract_even_without_handles():
    _assert_error(
        _program(ir.EvalStmt(_call("buffer.unregistered", []), SPAN)), "no registered buffer-stage contract"
    )


def test_functional_registered_operation_cannot_consume_buffer():
    buffer = _var("buffer")
    call = _call("tensor.read", [buffer, _int(0)], ir.ScalarType(DataType.FP32))
    _assert_error(_program(ir.EvalStmt(call, SPAN), [buffer]), "Non-buffer operation")


def test_unregistered_call_cannot_define_buffer():
    destination = _var("destination")
    call = _call("user_function", [], destination.type)
    _assert_error(_program(ir.AssignStmt(destination, call, SPAN)), "cannot produce a buffer handle")


@pytest.mark.parametrize("failure", ["arity", "result", "descriptor", "keyword", "attribute"])
def test_original_buffer_call_is_validated_without_repair(failure):
    source, destination = _var("source"), _var("destination")
    args = [source, destination]
    type_ = ir.VoidType()
    kwargs = None
    attrs = None
    if failure == "arity":
        args = [source]
    elif failure == "result":
        type_ = ir.ScalarType(DataType.INDEX)
    elif failure == "descriptor":
        destination = _var("destination", _buffer_type(8))
        args = [source, destination]
    elif failure == "keyword":
        kwargs = {"transpose": True}
    else:
        attrs = {"hidden_destination": destination}
    call = _call("buffer.copy", args, type_, kwargs, attrs)
    before = ir.serialize(call)
    _assert_error(
        _program(ir.EvalStmt(call, SPAN), [source, destination]),
        "Call attribute" if failure == "attribute" else "Invalid buffer call",
    )
    assert ir.serialize(call) == before


def _hidden_buffer_attr(source, form):
    if form == "var_list":
        return [source]
    if form == "expression":
        return ir.Add(source, _int(1), DataType.INDEX, SPAN)
    return source


@pytest.mark.parametrize("form", ["var", "var_list", "expression"])
def test_function_attributes_cannot_hide_buffer_references(form):
    source = _var("source")
    attrs = {"hidden_buffer": _hidden_buffer_attr(source, form)}
    _assert_error(_program(params=[source], attrs=attrs), "Function attribute 'hidden_buffer'")


@pytest.mark.parametrize("form", ["var", "var_list", "expression"])
@pytest.mark.parametrize("scope_type", [ir.InCoreScopeStmt, ir.RuntimeScopeStmt, ir.ClusterScopeStmt])
def test_scope_attributes_cannot_hide_buffer_references(scope_type, form):
    source = _var("source")
    carrier = ir.ForStmt(
        _var("i", ir.ScalarType(DataType.INDEX)),
        _int(0),
        _int(2),
        _int(1),
        [],
        ir.ReturnStmt(SPAN),
        [],
        SPAN,
        attrs={"hidden_buffer": _hidden_buffer_attr(source, form)},
    )
    # Scope attrs have no public mutation API. ForStmt supplies the same span,
    # body and attrs wire fields; scope readers ignore its extra loop fields.
    # Replace only the MessagePack node-kind string, without an optional Python
    # MessagePack dependency or manually encoding buffer/attribute descriptors.
    payload = bytes(ir.serialize(carrier))
    marker = b"\xa4type\xa7ForStmt"
    kind = scope_type.__name__.encode()
    assert len(kind) < 32 and payload.count(marker) == 1
    restored = ir.deserialize(payload.replace(marker, b"\xa4type" + bytes([0xA0 + len(kind)]) + kind))
    assert isinstance(restored, scope_type)
    _assert_error(_program(restored), "Scope attribute 'hidden_buffer'")


def test_for_attributes_cannot_hide_buffer_references():
    source = _var("source")
    loop = ir.ForStmt(
        _var("i", ir.ScalarType(DataType.INDEX)),
        _int(0),
        _int(2),
        _int(1),
        [],
        ir.YieldStmt(SPAN),
        [],
        SPAN,
        attrs={"hidden_buffer": [source]},
    )
    _assert_error(_program(loop, [source]), "For attribute 'hidden_buffer'")


def test_invalid_explicit_allocation_type_is_rejected():
    call = _allocation(ir.ScalarType(DataType.INDEX))
    _assert_error(_program(ir.EvalStmt(call, SPAN)), "Invalid buffer call")


@pytest.mark.parametrize(
    "failure,message",
    [
        ("missing_extent", "requires 1 runtime valid extent"),
        ("invalid_tuple", "must be a MakeTuple"),
        ("extent_out_of_bounds", "must be between 0 and 16"),
        ("negative_address", "effective address must be nonnegative"),
    ],
)
def test_original_allocation_operands_are_validated(failure, message):
    descriptor = _buffer_type()
    operands: list[ir.Expr] = [ir.MakeTuple([], SPAN)]
    if failure in {"missing_extent", "extent_out_of_bounds"}:
        descriptor = ir.BufferType([16, 32], DataType.FP32, ir.MemorySpace.Vec, valid_shape=[-1, 32])
        if failure == "extent_out_of_bounds":
            operands = [ir.MakeTuple([_int(17)], SPAN)]
    elif failure == "invalid_tuple":
        operands = [_int(0)]
    else:
        operands.append(_int(-1))
    destination = _var("destination", descriptor)
    call = _call("buffer.alloc", operands, descriptor)
    before = ir.serialize(call)
    _assert_error(_program(ir.AssignStmt(destination, call, SPAN)), message)
    assert ir.serialize(call) == before


def test_buffer_assignment_requires_explicit_alias_contract():
    source, alias = _var("source"), _var("alias")
    _assert_error(_program(ir.AssignStmt(alias, source, SPAN), [source]), "implicit alias")


def test_make_tuple_cannot_hide_buffer_aliases():
    source = _var("source")
    value = ir.MakeTuple([source], SPAN)
    _assert_error(_program(ir.EvalStmt(value, SPAN), [source]), "direct results of registered buffer")


@pytest.mark.parametrize("kind", ["if", "for", "while"])
@pytest.mark.parametrize("nested", [False, True])
def test_control_flow_buffer_slots_are_rejected(kind, nested):
    type_ = ir.TupleType([_buffer_type()]) if nested else _buffer_type()
    source, result = _var("source", type_), _var("result", type_)
    condition = ir.ConstBool(True, SPAN)
    if kind == "if":
        body = ir.IfStmt(
            condition, ir.YieldStmt([source], SPAN), ir.YieldStmt([source], SPAN), [result], SPAN
        )
    else:
        carry = ir.IterArg("carry", type_, source, SPAN)
        yield_stmt = ir.YieldStmt([carry], SPAN)
        if kind == "for":
            body = ir.ForStmt(
                _var("i", ir.ScalarType(DataType.INDEX)),
                _int(0),
                _int(2),
                _int(1),
                [carry],
                yield_stmt,
                [result],
                SPAN,
            )
        else:
            body = ir.WhileStmt(condition, [carry], yield_stmt, [result], SPAN)
    _assert_error(_program(body, [source]), "return_vars")


def test_buffer_yield_is_rejected_without_return_vars():
    source = _var("source")
    branch = ir.IfStmt(ir.ConstBool(True, SPAN), ir.YieldStmt([source], SPAN), None, [], SPAN)
    _assert_error(_program(branch, [source]), "Yield values")


def test_scalar_control_flow_can_write_outer_buffers():
    source, destination = _var("source"), _var("destination")
    carry = ir.IterArg("carry", ir.ScalarType(DataType.INDEX), _int(0), SPAN)
    result = _var("result", carry.type)
    body = ir.SeqStmts(
        [ir.EvalStmt(_call("buffer.copy", [source, destination]), SPAN), ir.YieldStmt([carry], SPAN)], SPAN
    )
    loop = ir.ForStmt(_var("i", carry.type), _int(0), _int(2), _int(1), [carry], body, [result], SPAN)
    assert _verify(_program(loop, [source, destination])) == []


def test_buffer_handles_cannot_escape_function():
    source = _var("source")
    _assert_error(_program(ir.ReturnStmt([source], SPAN), [source], [source.type]), "function return")


def test_submit_cannot_consume_device_buffer():
    source = _var("source")
    result_type = ir.TupleType([ir.ScalarType(DataType.TASK_ID)])
    submit = ir.Submit(ir.GlobalVar("callee"), [source], [], result_type, SPAN)
    _assert_error(_program(ir.EvalStmt(submit, SPAN), [source]), "Submit arguments")


def test_spmd_core_count_cannot_consume_device_buffer():
    source = _var("source")
    scope = ir.SpmdScopeStmt(source, False, "", ir.ReturnStmt(SPAN), SPAN)
    _assert_error(_program(scope, [source]), "SPMD core count cannot carry buffer handles")


def test_scalar_expression_cannot_consume_buffer():
    source = _var("source")
    expression = ir.Add(source, source, DataType.INDEX, SPAN)
    _assert_error(_program(ir.EvalStmt(expression, SPAN), [source]), "Scalar expression operand")


def test_use_before_definition_is_checked_by_existing_verifier():
    buffer = _var("missing")
    call = _call("buffer.mul", [buffer, buffer, buffer])
    _assert_error(_program(ir.EvalStmt(call, SPAN)), "missing", rule="UseAfterDefCheck")


def test_branch_local_allocation_cannot_escape_a_region_without_results():
    destination, local = _var("destination"), _var("local")
    branch = ir.IfStmt(
        ir.ConstBool(True, SPAN), ir.AssignStmt(local, _allocation(local.type), SPAN), None, [], SPAN
    )
    use = ir.EvalStmt(_call("buffer.copy", [local, destination]), SPAN)
    _assert_error(_program(ir.SeqStmts([branch, use], SPAN), [destination]), "local", rule="UseAfterDefCheck")


def test_assignment_type_symmetry_is_checked_by_existing_verifier():
    buffer = _var("buffer", _buffer_type(8))
    allocation = _allocation(_buffer_type())
    _assert_error(
        _program(ir.AssignStmt(buffer, allocation, SPAN)), "type mismatch", rule="AssignTypeSymmetry"
    )


def test_multiple_buffer_definitions_are_checked_by_existing_verifier():
    buffer = _var("buffer")
    first = ir.AssignStmt(buffer, _allocation(buffer.type), SPAN)
    second = ir.AssignStmt(buffer, _allocation(buffer.type), SPAN)
    diagnostics = _verify(_program(ir.SeqStmts([first, second], SPAN)))
    assert any(d.rule_name == "SSAVerify" for d in diagnostics)


def test_property_is_explicitly_selected():
    assert not passes.get_default_verify_properties().contains(passes.IRProperty.BufferIR)
    assert not passes.get_structural_properties().contains(passes.IRProperty.BufferIR)
    assert "BufferIR" in str(passes.IRProperty.BufferIR)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
