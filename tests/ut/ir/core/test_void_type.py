# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Void calls execute as statements and never stand in for SSA values."""

import pytest
from pypto import DataType, ir


@pytest.fixture
def span() -> ir.Span:
    return ir.Span("void_test.py", 7, 3, 7, 12)


@pytest.fixture
def void_call(span: ir.Span) -> ir.Call:
    return ir.Call(ir.Op("test.void"), [], ir.VoidType.get(), span)


def test_void_call_can_execute_without_binding_a_value(void_call: ir.Call, span: ir.Span):
    stmt = ir.EvalStmt(void_call, span)
    assert isinstance(stmt.expr.type, ir.VoidType)
    assert not isinstance(stmt.expr.type, ir.UnknownType)
    func = ir.Function("execute", [], [], stmt, span)
    assert list(func.return_types) == []
    assert list(ir.YieldStmt([], span).value) == []
    assert list(ir.ReturnStmt([], span).value) == []


def test_void_cannot_be_a_variable_type(span: ir.Span):
    with pytest.raises(ValueError, match="Var type.*VoidType") as error:
        ir.Var("missing", ir.VoidType(), span)
    assert "void_test.py" in str(error.value)


def test_void_cannot_be_a_loop_carry_type(span: ir.Span):
    init = ir.ConstInt(0, DataType.INT64, span)
    with pytest.raises(ValueError, match="Var type.*VoidType"):
        ir.IterArg("missing", ir.VoidType(), init, span)


def test_void_cannot_initialize_a_loop_carry(void_call: ir.Call, span: ir.Span):
    with pytest.raises(ValueError, match="IterArg initial value.*VoidType"):
        ir.IterArg("acc", ir.ScalarType(DataType.INT64), void_call, span)


def test_void_cannot_be_assigned_even_to_unknown_type(void_call: ir.Call, span: ir.Span):
    var = ir.Var("unknown", ir.UnknownType(), span)
    with pytest.raises(ValueError, match="AssignStmt value.*VoidType"):
        ir.AssignStmt(var, void_call, span)


@pytest.mark.parametrize("statement", [ir.YieldStmt, ir.ReturnStmt])
def test_void_cannot_be_transferred_by_control_flow(statement, void_call: ir.Call, span: ir.Span):
    with pytest.raises(ValueError, match="VoidType"):
        statement([ir.ConstInt(1, DataType.INT64, span), void_call], span)


@pytest.mark.parametrize("constructor", ["basic", "typed", "kwargs", "kwargs_typed", "attrs"])
def test_every_call_constructor_rejects_void_arguments(constructor: str, void_call: ir.Call, span: ir.Span):
    op = ir.Op("test.consumer")
    result_type = ir.ScalarType(DataType.INT64)
    with pytest.raises(ValueError, match="Call argument.*VoidType"):
        if constructor == "basic":
            ir.Call(op, [void_call], span)
        elif constructor == "typed":
            ir.Call(op, [void_call], result_type, span)
        elif constructor == "kwargs":
            ir.Call(op, [void_call], {}, span)
        elif constructor == "kwargs_typed":
            ir.Call(op, [void_call], {}, result_type, span)
        else:
            ir.Call(op, [void_call], {}, {}, result_type, span)


def _call_with_metadata(constructor: str, metadata: dict[str, object], result_type: ir.Type, span: ir.Span):
    op = ir.Op("test.launch")
    if constructor == "kwargs":
        return ir.Call(op, [], metadata, span)
    if constructor == "kwargs_typed":
        return ir.Call(op, [], metadata, result_type, span)
    if constructor == "kwargs_attrs":
        return ir.Call(op, [], metadata, {}, result_type, span)
    if constructor == "attrs":
        return ir.Call(op, [], {}, metadata, result_type, span)
    return ir.set_call_attrs(ir.Call(op, [], result_type, span), metadata)


@pytest.mark.parametrize("constructor", ["kwargs", "kwargs_typed", "kwargs_attrs", "attrs", "set_attrs"])
@pytest.mark.parametrize("key", ["core_num", "predicate", "custom_expr"])
@pytest.mark.parametrize("explicit_span", [False, True])
def test_call_metadata_rejects_void_expressions(constructor, key, explicit_span, void_call, span):
    """All metadata construction paths reject void values and retain a useful location."""
    location = span if explicit_span else ir.Span.unknown()
    context = "Call keyword argument" if constructor.startswith("kwargs") else "Call attribute"
    with pytest.raises(ValueError, match=f"{context} '{key}'.*VoidType") as error:
        _call_with_metadata(constructor, {key: void_call}, ir.ScalarType(DataType.INT64), location)
    assert "void_test.py" in str(error.value)


@pytest.mark.parametrize("constructor", ["kwargs", "kwargs_typed", "kwargs_attrs", "attrs", "set_attrs"])
@pytest.mark.parametrize("result_type", [ir.ScalarType(DataType.INT64), ir.UnknownType(), ir.VoidType()])
def test_call_metadata_accepts_value_expressions_and_plain_metadata(constructor, result_type, span):
    """Value metadata stays valid even for a call with no result, including after serialization."""
    var = ir.Var("task", ir.ScalarType(DataType.INDEX), span)
    metadata = {
        "core_num": ir.Add(var, ir.ConstInt(1, DataType.INDEX, span), DataType.INDEX, span),
        "predicate": ir.ConstBool(True, span),
        "custom_expr": ir.Call(ir.Op("test.unknown"), [], span),
        "task_id_var": var,
        "dump_vars": [var],
        "arg_direction_overrides": [],
        "enabled": True,
        "label": "launch",
        "dtype": DataType.FP32,
    }
    call = _call_with_metadata(constructor, metadata, result_type, span)
    ir.assert_structural_equal(call, ir.deserialize(ir.serialize(call)), enable_auto_mapping=True)


def test_void_cannot_be_a_tuple_element(void_call: ir.Call, span: ir.Span):
    with pytest.raises(ValueError, match="MakeTuple element.*VoidType"):
        ir.MakeTuple([void_call], span)
    with pytest.raises(ValueError, match="TupleType element.*VoidType"):
        ir.TupleType([ir.ScalarType(DataType.INT64), ir.VoidType()])


def test_void_cannot_be_a_function_result_entry(void_call: ir.Call, span: ir.Span):
    with pytest.raises(ValueError, match="Function return type.*VoidType"):
        ir.Function("invalid", [], [ir.VoidType()], ir.EvalStmt(void_call, span), span)


@pytest.mark.parametrize("type_class", [ir.TensorType, ir.TileType])
def test_void_cannot_be_hidden_in_a_shape_dimension(type_class, void_call: ir.Call):
    with pytest.raises(ValueError, match="ShapedType dimension.*VoidType"):
        type_class([void_call], DataType.FP32)


@pytest.mark.parametrize("field", ["stride", "valid_shape"])
def test_void_cannot_be_hidden_in_a_tensor_view(field: str, void_call: ir.Call):
    stride = [void_call] if field == "stride" else []
    valid = [void_call] if field == "valid_shape" else []
    with pytest.raises(ValueError, match="TensorView.*VoidType"):
        ir.TensorView(stride, ir.TensorLayout.ND, valid)

    view = ir.TensorView()
    setattr(view, field, [void_call])
    with pytest.raises(ValueError, match="TensorView.*VoidType"):
        ir.TensorType([32], DataType.FP32, tensor_view=view)


@pytest.mark.parametrize("field", ["stride", "valid_shape", "start_offset"])
def test_void_cannot_be_hidden_in_a_tile_view(field: str, void_call: ir.Call, span: ir.Span):
    stride = [void_call] if field == "stride" else []
    valid = [void_call] if field == "valid_shape" else []
    offset = void_call if field == "start_offset" else ir.ConstInt(0, DataType.INDEX, span)
    with pytest.raises(ValueError, match="TileView.*VoidType"):
        ir.TileView(valid, stride, offset)


def test_void_cannot_be_hidden_in_memref_addressing(void_call: ir.Call, span: ir.Span):
    base = ir.Var("base", ir.PtrType(), span)
    with pytest.raises(ValueError, match="MemRef byte_offset.*VoidType"):
        ir.MemRef(base, void_call, 128, span)
    with pytest.raises(ValueError, match="MemRef slot_index.*VoidType"):
        ir.MemRef(base, 0, 128, span, slots=2, slot=void_call)


@pytest.mark.parametrize("explicit_span", [False, True])
def test_void_cannot_size_a_window_buffer(void_call: ir.Call, span: ir.Span, explicit_span: bool):
    """Window allocation sizes must produce a value, with a useful source location."""
    base = ir.Var("base", ir.PtrType(), span)
    with pytest.raises(ValueError, match="WindowBuffer size.*VoidType") as error:
        if explicit_span:
            ir.WindowBuffer(base, void_call, span=span)
        else:
            ir.WindowBuffer(base, void_call)
    assert "void_test.py" in str(error.value)


def test_void_cannot_be_wrapped_in_scalar_expressions(void_call: ir.Call, span: ir.Span):
    scalar = ir.ConstInt(1, DataType.INT64, span)
    with pytest.raises(ValueError, match="BinaryExpr left operand.*VoidType"):
        ir.Add(void_call, scalar, DataType.INT64, span)
    with pytest.raises(ValueError, match="BinaryExpr right operand.*VoidType"):
        ir.Add(scalar, void_call, DataType.INT64, span)
    with pytest.raises(ValueError, match="UnaryExpr operand.*VoidType"):
        ir.Cast(void_call, DataType.INT64, span)


def test_void_cannot_control_a_branch_or_loop(void_call: ir.Call, span: ir.Span):
    body = ir.YieldStmt([], span)
    with pytest.raises(ValueError, match="IfStmt condition.*VoidType"):
        ir.IfStmt(void_call, body, None, [], span)
    with pytest.raises(ValueError, match="WhileStmt condition.*VoidType"):
        ir.WhileStmt(void_call, [], body, [], span)


@pytest.mark.parametrize("position", [0, 1, 2])
def test_void_cannot_be_a_for_loop_bound(position: int, void_call: ir.Call, span: ir.Span):
    loop_var = ir.Var("i", ir.ScalarType(DataType.INDEX), span)
    bounds: list[ir.Expr] = [
        ir.ConstInt(0, DataType.INDEX, span),
        ir.ConstInt(8, DataType.INDEX, span),
        ir.ConstInt(1, DataType.INDEX, span),
    ]
    bounds[position] = void_call
    with pytest.raises(ValueError, match="ForStmt.*VoidType"):
        ir.ForStmt(loop_var, bounds[0], bounds[1], bounds[2], [], ir.YieldStmt([], span), [], span)


@pytest.mark.parametrize("field", ["argument", "dependency", "result"])
def test_submit_rejects_void_values(field: str, void_call: ir.Call, span: ir.Span):
    args = [void_call] if field == "argument" else []
    deps = [void_call] if field == "dependency" else []
    result_type = ir.VoidType() if field == "result" else ir.ScalarType(DataType.TASK_ID)
    with pytest.raises(ValueError, match="Submit.*VoidType"):
        ir.Submit(ir.GlobalVar("kernel"), args, deps, result_type, span)


def test_unknown_values_keep_their_existing_construction_contract(span: ir.Span):
    unknown = ir.Call(ir.Op("test.unresolved"), [], span)
    var = ir.Var("value", ir.UnknownType(), span)
    assert isinstance(ir.AssignStmt(var, unknown, span).value.type, ir.UnknownType)
    assert len(ir.YieldStmt([unknown], span).value) == 1
    assert len(ir.ReturnStmt([unknown], span).value) == 1
    assert isinstance(ir.Call(ir.Op("test.consumer"), [unknown], span).type, ir.UnknownType)
    assert isinstance(ir.MakeTuple([unknown], span).type, ir.TupleType)
    assert isinstance(ir.IterArg("carry", ir.UnknownType(), unknown, span).type, ir.UnknownType)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
