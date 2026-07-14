# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for valid-shape rank and bounds verification."""

import pytest
from pypto import DataType, ir, passes

_SPAN = ir.Span.unknown()


def _const(value: int) -> ir.ConstInt:
    return ir.ConstInt(value, DataType.INDEX, _SPAN)


def _sym(name: str) -> ir.Var:
    return ir.Var(name, ir.ScalarType(DataType.INDEX), _SPAN)


def _add(lhs: ir.Expr, rhs: ir.Expr) -> ir.Add:
    return ir.Add(lhs, rhs, DataType.INDEX, _SPAN)


def _sub(lhs: ir.Expr, rhs: ir.Expr) -> ir.Sub:
    return ir.Sub(lhs, rhs, DataType.INDEX, _SPAN)


def _tensor_type(shape: list[ir.Expr], valid_shape: list[ir.Expr] | None = None) -> ir.TensorType:
    view = None if valid_shape is None else ir.TensorView(layout=ir.TensorLayout.ND, valid_shape=valid_shape)
    return ir.TensorType(shape, DataType.FP32, None, view)


def _tile_type(shape: list[ir.Expr], valid_shape: list[ir.Expr] | None = None) -> ir.TileType:
    view = None if valid_shape is None else ir.TileView(valid_shape=valid_shape)
    return ir.TileType(shape, DataType.FP32, None, view)


def _program(
    *,
    params: list[ir.Var] | None = None,
    return_types: list[ir.Type] | None = None,
    statements: list[ir.Stmt] | None = None,
) -> ir.Program:
    body = ir.SeqStmts([*(statements or []), ir.ReturnStmt([], _SPAN)], _SPAN)
    function = ir.Function("main", params or [], return_types or [], body, _SPAN)
    return ir.Program([function], "test", _SPAN)


def _verify(program: ir.Program) -> list[passes.Diagnostic]:
    properties = passes.IRPropertySet()
    properties.insert(passes.IRProperty.TypeChecked)
    return passes.PropertyVerifierRegistry.verify(properties, program)


def test_rejects_negative_tensor_valid_extent():
    invalid_type = _tensor_type([_const(16)], [_const(-1)])
    diagnostics = _verify(_program(params=[ir.Var("x", invalid_type, _SPAN)]))

    assert len(diagnostics) == 1
    assert diagnostics[0].error_code == 104
    assert "TensorType valid_shape dimension 0" in diagnostics[0].message
    assert "negative extent -1" in diagnostics[0].message


def test_rejects_tile_valid_extent_larger_than_shape():
    invalid_type = _tile_type([_const(8), _const(16)], [_const(8), _const(17)])
    diagnostics = _verify(_program(return_types=[invalid_type]))

    assert len(diagnostics) == 1
    assert diagnostics[0].error_code == 104
    assert "Function 'main' return type[0]" in diagnostics[0].message
    assert "TileType valid_shape dimension 1 extent 17" in diagnostics[0].message
    assert "physical shape extent 16" in diagnostics[0].message


def test_rejects_valid_shape_rank_mismatch():
    invalid_type = _tensor_type([_const(8), _const(16)], [_const(8)])
    diagnostics = _verify(_program(params=[ir.Var("x", invalid_type, _SPAN)]))

    assert len(diagnostics) == 1
    assert diagnostics[0].error_code == 103
    assert "TensorType valid_shape rank mismatch" in diagnostics[0].message
    assert "got rank 1" in diagnostics[0].message
    assert "physical shape has rank 2" in diagnostics[0].message


def test_unset_and_explicit_full_valid_shapes_pass():
    unset_tensor = _tensor_type([_const(8), _const(16)])
    full_tensor = _tensor_type([_const(8), _const(16)], [_const(8), _const(16)])
    unset_tile = _tile_type([_const(8), _const(16)])
    full_tile = _tile_type([_const(8), _const(16)], [_const(8), _const(16)])

    assert full_tensor.tensor_view is None
    assert full_tile.tile_view is None
    params = [
        ir.Var("unset_tensor", unset_tensor, _SPAN),
        ir.Var("full_tensor", full_tensor, _SPAN),
        ir.Var("unset_tile", unset_tile, _SPAN),
        ir.Var("full_tile", full_tile, _SPAN),
    ]
    assert _verify(_program(params=params)) == []


def test_allows_genuinely_unknown_symbolic_bounds():
    valid_rows = _sym("valid_rows")
    symbolic_type = _tensor_type([_const(16)], [valid_rows])

    assert _verify(_program(params=[ir.Var("x", symbolic_type, _SPAN)])) == []


def test_allows_unsupported_extent_type_as_unknown():
    float_extent = ir.ConstFloat(1.0, DataType.FP32, _SPAN)
    unsupported_type = _tensor_type([_const(16)], [float_extent])

    assert _verify(_program(params=[ir.Var("x", unsupported_type, _SPAN)])) == []


def test_analyzer_proves_composite_extent_in_bounds():
    n = _sym("n")
    valid = _sub(_add(n, _const(64)), n)
    symbolic_type = _tensor_type([_const(64)], [valid])

    assert _verify(_program(params=[ir.Var("x", symbolic_type, _SPAN)])) == []


def test_analyzer_rejects_composite_extent_larger_than_shape():
    n = _sym("n")
    valid = _sub(_add(n, _const(65)), n)
    invalid_type = _tensor_type([_const(64)], [valid])
    diagnostics = _verify(_program(params=[ir.Var("x", invalid_type, _SPAN)]))

    assert len(diagnostics) == 1
    assert "TensorType valid_shape dimension 0" in diagnostics[0].message
    assert "65" in diagnostics[0].message
    assert "physical shape extent 64" in diagnostics[0].message


def test_analyzer_rejects_composite_negative_extent():
    n = _sym("n")
    valid = _sub(_sub(n, n), _const(1))
    invalid_type = _tensor_type([_const(64)], [valid])
    diagnostics = _verify(_program(params=[ir.Var("x", invalid_type, _SPAN)]))

    assert len(diagnostics) == 1
    assert "provably negative extent" in diagnostics[0].message
    assert "n" in diagnostics[0].message


def test_recurses_into_tuple_function_return_type():
    invalid_tensor = _tensor_type([_const(8)], [_const(9)])
    tuple_type = ir.TupleType([ir.ScalarType(DataType.INT64), invalid_tensor])
    diagnostics = _verify(_program(return_types=[tuple_type]))

    assert len(diagnostics) == 1
    assert "return type[0] TupleType element[1]" in diagnostics[0].message
    assert "TensorType valid_shape dimension 0" in diagnostics[0].message


def test_checks_distributed_tensor_valid_shape():
    view = ir.TensorView(layout=ir.TensorLayout.ND, valid_shape=[_const(4), _const(17)])
    invalid_type = ir.DistributedTensorType([_const(4), _const(16)], DataType.FP32, None, view)
    diagnostics = _verify(_program(params=[ir.Var("remote", invalid_type, _SPAN)]))

    assert len(diagnostics) == 1
    assert "DistributedTensorType valid_shape dimension 1" in diagnostics[0].message


def test_checks_local_variable_type():
    invalid_type = _tile_type([_const(16)], [_const(17)])
    local = ir.Var("local", invalid_type, _SPAN)
    assign = ir.AssignStmt(local, ir.ConstInt(0, DataType.INT64, _SPAN), _SPAN)
    diagnostics = _verify(_program(statements=[assign]))

    assert len(diagnostics) == 1
    assert "Var 'local' has invalid TileType valid_shape dimension 0" in diagnostics[0].message


def test_checks_iter_arg_type():
    invalid_type = _tensor_type([_const(16)], [_const(17)])
    scalar_type = ir.ScalarType(DataType.INT64)
    initial = ir.Var("initial", scalar_type, _SPAN)
    carry = ir.IterArg("carry", invalid_type, initial, _SPAN)
    result = ir.Var("result", scalar_type, _SPAN)
    loop = ir.ForStmt(
        ir.Var("i", ir.ScalarType(DataType.INDEX), _SPAN),
        _const(0),
        _const(1),
        _const(1),
        [carry],
        ir.YieldStmt([initial], _SPAN),
        [result],
        _SPAN,
    )
    diagnostics = _verify(_program(statements=[loop]))

    bounds_diagnostics = [d for d in diagnostics if "has invalid" in d.message]
    assert len(bounds_diagnostics) == 1
    assert "IterArg 'carry' has invalid TensorType" in bounds_diagnostics[0].message


def test_checks_call_result_type():
    invalid_type = _tensor_type([_const(16)], [_const(17)])
    call = ir.Call(ir.Op("test.call"), [], invalid_type, _SPAN)
    diagnostics = _verify(_program(statements=[ir.EvalStmt(call, _SPAN)]))

    assert len(diagnostics) == 1
    assert "Call result has invalid TensorType valid_shape dimension 0" in diagnostics[0].message


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
