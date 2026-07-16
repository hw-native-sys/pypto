# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Positive and negative contracts for the PTOBufferized verifier."""

import pytest
from pypto.pypto_core import DataType, ir, passes


def _span() -> ir.Span:
    return ir.Span.unknown()


def _buf_type() -> ir.PTOTileBufType:
    return ir.PTOTileBufType(ir.MemorySpace.Vec, DataType.FP32, 32, 32)


def _index(value: int) -> ir.ConstInt:
    return ir.ConstInt(value, DataType.INDEX, _span())


def _call(op_name: str, args: list[ir.Expr], result_type: ir.Type) -> ir.Call:
    return ir.Call(ir.get_op(op_name), args, result_type, _span())


def _alloc(name: str, result_type: ir.Type | None = None) -> tuple[ir.Var, ir.AssignStmt]:
    result_type = result_type or _buf_type()
    var = ir.Var(name, result_type, _span())
    call = _call("pto.alloc_tile", [_index(32), _index(32)], result_type)
    return var, ir.AssignStmt(var, call, _span())


def _program(
    statements: list[ir.Stmt],
    params: list[ir.Var] | None = None,
    function_type: ir.FunctionType = ir.FunctionType.InCore,
) -> ir.Program:
    params = params or []
    body = ir.SeqStmts([*statements, ir.ReturnStmt([], _span())], _span())
    function = ir.Function(
        "kernel",
        [(param, ir.ParamDirection.In) for param in params],
        [],
        body,
        _span(),
        function_type,
    )
    return ir.Program([function], "pto_target_test", _span())


def _verify(program: ir.Program) -> list[passes.Diagnostic]:
    properties = passes.IRPropertySet()
    properties.insert(passes.IRProperty.PTOBufferized)
    return passes.PropertyVerifierRegistry.verify(properties, program)


def _messages(program: ir.Program) -> list[str]:
    return [diagnostic.message for diagnostic in _verify(program)]


def test_valid_hand_built_pto_target_ir_passes():
    lhs, lhs_alloc = _alloc("lhs")
    rhs, rhs_alloc = _alloc("rhs")
    out, out_alloc = _alloc("out")
    tadd = _call("pto.tadd", [lhs, rhs, out], ir.UnknownType.get())

    assert _verify(_program([lhs_alloc, rhs_alloc, out_alloc, ir.EvalStmt(tadd, _span())])) == []


def test_tadd_missing_output_is_rejected():
    lhs = ir.Var("lhs", _buf_type(), _span())
    rhs = ir.Var("rhs", _buf_type(), _span())
    tadd = _call("pto.tadd", [lhs, rhs], ir.UnknownType.get())

    messages = _messages(_program([ir.EvalStmt(tadd, _span())]))
    assert any("invalid operand count 2" in message for message in messages)


def test_target_handle_use_must_be_dominated_by_allocation():
    lhs = ir.Var("lhs", _buf_type(), _span())
    rhs, rhs_alloc = _alloc("rhs")
    out, out_alloc = _alloc("out")
    tadd = _call("pto.tadd", [lhs, rhs, out], ir.UnknownType.get())

    messages = _messages(_program([rhs_alloc, out_alloc, ir.EvalStmt(tadd, _span())]))
    assert any("has no dominating allocation" in message for message in messages)


def test_tadd_output_must_be_tile_buffer():
    lhs = ir.Var("lhs", _buf_type(), _span())
    rhs = ir.Var("rhs", _buf_type(), _span())
    scalar_out = ir.Var("out", ir.ScalarType(DataType.FP32), _span())
    tadd = _call("pto.tadd", [lhs, rhs, scalar_out], ir.UnknownType.get())

    messages = _messages(_program([ir.EvalStmt(tadd, _span())]))
    assert any("operand #2 must have PTOTileBufType" in message for message in messages)


def test_tadds_scalar_operand_must_have_scalar_type():
    source, source_alloc = _alloc("source")
    not_scalar, not_scalar_alloc = _alloc("not_scalar")
    out, out_alloc = _alloc("out")
    tadds = _call("pto.tadds", [source, not_scalar, out], ir.UnknownType.get())

    messages = _messages(_program([source_alloc, not_scalar_alloc, out_alloc, ir.EvalStmt(tadds, _span())]))
    assert any("operand #1 must have ScalarType" in message for message in messages)


def test_alloc_result_must_be_tile_buffer():
    scalar_type = ir.ScalarType(DataType.INT32)
    _, bad_alloc = _alloc("bad", scalar_type)
    messages = _messages(_program([bad_alloc]))

    assert any("result must have PTOTileBufType" in message for message in messages)
    assert any("allocation destination must have PTOTileBufType" in message for message in messages)


def test_destination_passing_op_must_be_eval_stmt():
    lhs = ir.Var("lhs", _buf_type(), _span())
    rhs = ir.Var("rhs", _buf_type(), _span())
    out = ir.Var("out", _buf_type(), _span())
    fake_result = ir.Var("result", ir.UnknownType.get(), _span())
    tadd = _call("pto.tadd", [lhs, rhs, out], ir.UnknownType.get())

    messages = _messages(_program([ir.AssignStmt(fake_result, tadd, _span())]))
    assert any("must appear in an EvalStmt" in message for message in messages)


def test_alloc_must_be_assign_stmt():
    call = _call("pto.alloc_tile", [_index(32), _index(32)], _buf_type())

    messages = _messages(_program([ir.EvalStmt(call, _span())]))
    assert any("must be the value of an AssignStmt" in message for message in messages)


def test_destination_passing_op_cannot_return_value():
    lhs = ir.Var("lhs", _buf_type(), _span())
    rhs = ir.Var("rhs", _buf_type(), _span())
    out = ir.Var("out", _buf_type(), _span())
    tadd = _call("pto.tadd", [lhs, rhs, out], _buf_type())

    messages = _messages(_program([ir.EvalStmt(tadd, _span())]))
    assert any("must not return a value" in message for message in messages)


def test_pto_target_op_is_rejected_in_orchestration_function():
    _, alloc = _alloc("buf")

    messages = _messages(_program([alloc], function_type=ir.FunctionType.Orchestration))
    assert any("cannot appear in an Orchestration function" in message for message in messages)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
