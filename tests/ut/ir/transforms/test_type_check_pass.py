# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for TypeCheck pass."""

from pypto import ir
from pypto.pypto_core import DataType, passes


def test_type_check_for_type_mismatch():
    """Test TypeCheck detects type mismatch in ForStmt."""
    span = ir.Span.unknown()

    a = ir.Var("a", ir.ScalarType(DataType.INT64), span)
    params: list[ir.Var] = [a]
    return_types: list[ir.Type] = [ir.ScalarType(DataType.INT64)]

    loop_var = ir.Var("i", ir.ScalarType(DataType.INT64), span)
    iter_arg = ir.IterArg("sum", ir.ScalarType(DataType.INT64), a, span)  # INT64
    yield_value = ir.ConstFloat(1.0, DataType.FP32, span)  # FP32 - mismatch!
    body = ir.YieldStmt([yield_value], span)
    result_var = ir.Var("result", ir.ScalarType(DataType.INT64), span)

    for_stmt = ir.ForStmt(
        loop_var,
        ir.ConstInt(0, DataType.INT64, span),
        ir.ConstInt(10, DataType.INT64, span),
        ir.ConstInt(1, DataType.INT64, span),
        [iter_arg],
        body,
        [result_var],
        span,
    )

    func_body = ir.SeqStmts([for_stmt, ir.ReturnStmt([result_var], span)], span)
    func = ir.Function("test_type_mismatch", params, return_types, func_body, span)
    program = ir.Program([func], "test_program", span)

    # Run type checking - should log type mismatch errors
    type_check_pass = passes.type_check()
    result_program = type_check_pass(program)
    assert result_program is not None


def test_type_check_if_type_mismatch():
    """Test TypeCheck detects type mismatch in IfStmt."""
    span = ir.Span.unknown()

    a = ir.Var("a", ir.ScalarType(DataType.INT64), span)
    params: list[ir.Var] = [a]
    return_types: list[ir.Type] = [ir.ScalarType(DataType.INT64)]

    condition = ir.Gt(a, ir.ConstInt(0, DataType.INT64, span), DataType.BOOL, span)
    then_body = ir.YieldStmt([ir.ConstInt(1, DataType.INT64, span)], span)  # INT64
    else_body = ir.YieldStmt([ir.ConstFloat(2.0, DataType.FP32, span)], span)  # FP32 - mismatch!
    result_var = ir.Var("result", ir.ScalarType(DataType.INT64), span)

    if_stmt = ir.IfStmt(condition, then_body, else_body, [result_var], span)

    func_body = ir.SeqStmts([if_stmt, ir.ReturnStmt([result_var], span)], span)
    func = ir.Function("test_if_type_mismatch", params, return_types, func_body, span)
    program = ir.Program([func], "test_program", span)

    # Run type checking - should log type mismatch error
    type_check_pass = passes.type_check()
    result_program = type_check_pass(program)
    assert result_program is not None


def test_type_check_tensor_shape_mismatch():
    """Test TypeCheck detects shape mismatch in TensorType."""
    span = ir.Span.unknown()

    shape1 = [ir.ConstInt(10, DataType.INT64, span), ir.ConstInt(20, DataType.INT64, span)]
    shape2 = [ir.ConstInt(10, DataType.INT64, span), ir.ConstInt(30, DataType.INT64, span)]  # Different!

    tensor_type1 = ir.TensorType(shape1, DataType.FP32)
    tensor_type2 = ir.TensorType(shape2, DataType.FP32)

    a = ir.Var("a", tensor_type1, span)
    params: list[ir.Var] = [a]
    return_types: list[ir.Type] = [tensor_type1]

    loop_var = ir.Var("i", ir.ScalarType(DataType.INT64), span)
    iter_arg = ir.IterArg("tensor", tensor_type1, a, span)
    yield_value = ir.Var("temp", tensor_type2, span)
    body = ir.YieldStmt([yield_value], span)
    result_var = ir.Var("result", tensor_type1, span)

    for_stmt = ir.ForStmt(
        loop_var,
        ir.ConstInt(0, DataType.INT64, span),
        ir.ConstInt(10, DataType.INT64, span),
        ir.ConstInt(1, DataType.INT64, span),
        [iter_arg],
        body,
        [result_var],
        span,
    )

    func_body = ir.SeqStmts([for_stmt, ir.ReturnStmt([result_var], span)], span)
    func = ir.Function("test_shape_mismatch", params, return_types, func_body, span)
    program = ir.Program([func], "test_program", span)

    # Run type checking - should log shape mismatch error
    type_check_pass = passes.type_check()
    result_program = type_check_pass(program)
    assert result_program is not None


def test_type_check_dimension_count_mismatch():
    """Test TypeCheck detects dimension count mismatch."""
    span = ir.Span.unknown()

    shape1 = [ir.ConstInt(10, DataType.INT64, span), ir.ConstInt(20, DataType.INT64, span)]
    shape2 = [ir.ConstInt(10, DataType.INT64, span)]  # Only 1 dimension!

    tensor_type1 = ir.TensorType(shape1, DataType.FP32)
    tensor_type2 = ir.TensorType(shape2, DataType.FP32)

    a = ir.Var("a", tensor_type1, span)
    params: list[ir.Var] = [a]
    return_types: list[ir.Type] = [tensor_type1]

    condition = ir.Gt(
        ir.ConstInt(1, DataType.INT64, span), ir.ConstInt(0, DataType.INT64, span), DataType.BOOL, span
    )
    then_body = ir.YieldStmt([ir.Var("t1", tensor_type1, span)], span)
    else_body = ir.YieldStmt([ir.Var("t2", tensor_type2, span)], span)  # Different dimensions!
    result_var = ir.Var("result", tensor_type1, span)

    if_stmt = ir.IfStmt(condition, then_body, else_body, [result_var], span)

    func_body = ir.SeqStmts([if_stmt, ir.ReturnStmt([result_var], span)], span)
    func = ir.Function("test_dim_mismatch", params, return_types, func_body, span)
    program = ir.Program([func], "test_program", span)

    # Run type checking - should log dimension mismatch error
    type_check_pass = passes.type_check()
    result_program = type_check_pass(program)
    assert result_program is not None


def test_type_check_tile_shape_mismatch():
    """Test TypeCheck detects shape mismatch in TileType."""
    span = ir.Span.unknown()

    shape1 = [ir.ConstInt(16, DataType.INT64, span), ir.ConstInt(16, DataType.INT64, span)]
    shape2 = [ir.ConstInt(16, DataType.INT64, span), ir.ConstInt(32, DataType.INT64, span)]  # Different!

    tile_type1 = ir.TileType(shape1, DataType.FP16)
    tile_type2 = ir.TileType(shape2, DataType.FP16)

    a = ir.Var("a", tile_type1, span)
    params: list[ir.Var] = [a]
    return_types: list[ir.Type] = [tile_type1]

    condition = ir.Gt(
        ir.ConstInt(1, DataType.INT64, span), ir.ConstInt(0, DataType.INT64, span), DataType.BOOL, span
    )
    then_body = ir.YieldStmt([ir.Var("tile1", tile_type1, span)], span)
    else_body = ir.YieldStmt([ir.Var("tile2", tile_type2, span)], span)  # Different shape!
    result_var = ir.Var("result", tile_type1, span)

    if_stmt = ir.IfStmt(condition, then_body, else_body, [result_var], span)

    func_body = ir.SeqStmts([if_stmt, ir.ReturnStmt([result_var], span)], span)
    func = ir.Function(
        "test_tile_shape_mismatch", params, return_types, func_body, span, ir.FunctionType.InCore
    )
    program = ir.Program([func], "test_program", span)

    # Run type checking - should log shape mismatch error
    type_check_pass = passes.type_check()
    result_program = type_check_pass(program)
    assert result_program is not None


def test_type_check_valid_types():
    """Test valid types pass type checking."""
    span = ir.Span.unknown()

    shape = [ir.ConstInt(10, DataType.INT64, span), ir.ConstInt(20, DataType.INT64, span)]
    tensor_type = ir.TensorType(shape, DataType.FP32)

    a = ir.Var("a", tensor_type, span)
    params: list[ir.Var] = [a]
    return_types: list[ir.Type] = [tensor_type]

    loop_var = ir.Var("i", ir.ScalarType(DataType.INT64), span)
    iter_arg = ir.IterArg("tensor", tensor_type, a, span)
    yield_value = ir.Var("temp", tensor_type, span)
    body = ir.YieldStmt([yield_value], span)
    result_var = ir.Var("result", tensor_type, span)

    for_stmt = ir.ForStmt(
        loop_var,
        ir.ConstInt(0, DataType.INT64, span),
        ir.ConstInt(10, DataType.INT64, span),
        ir.ConstInt(1, DataType.INT64, span),
        [iter_arg],
        body,
        [result_var],
        span,
    )

    func_body = ir.SeqStmts([for_stmt, ir.ReturnStmt([result_var], span)], span)
    func = ir.Function("test_valid", params, return_types, func_body, span)
    program = ir.Program([func], "test_program", span)

    # Run type checking - should pass without errors
    type_check_pass = passes.type_check()
    result_program = type_check_pass(program)
    assert result_program is not None


if __name__ == "__main__":
    test_type_check_for_type_mismatch()
    test_type_check_if_type_mismatch()
    test_type_check_tensor_shape_mismatch()
    test_type_check_dimension_count_mismatch()
    test_type_check_tile_shape_mismatch()
    test_type_check_valid_types()
