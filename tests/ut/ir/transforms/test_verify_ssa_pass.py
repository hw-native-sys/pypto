# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for VerifySSA pass (factory function style)."""

from pypto import ir
from pypto.ir import builder
from pypto.pypto_core import DataType, passes


def test_verify_ssa_valid():
    """Test VerifySSA with valid SSA IR."""
    ib = builder.IRBuilder()

    with ib.function("test_valid_ssa") as f:
        a = f.param("a", ir.ScalarType(DataType.INT64))
        b = f.param("b", ir.ScalarType(DataType.INT64))
        f.return_type(ir.ScalarType(DataType.INT64))

        x = ib.let("x", a)
        _y = ib.let("y", b)
        z = ib.let("z", x)

        ib.return_stmt(z)

    func = f.get_result()
    program = ir.Program([func], "test_program", ir.Span.unknown())

    # Run verification using factory function
    verify_pass = passes.verify_ssa()
    result_program = verify_pass(program)

    assert result_program is not None


def test_verify_ssa_multiple_assignment():
    """Test VerifySSA detects multiple assignments."""
    ib = builder.IRBuilder()

    with ib.function("test_multiple_assignment") as f:
        a = f.param("a", ir.ScalarType(DataType.INT64))
        f.return_type(ir.ScalarType(DataType.INT64))

        _x = ib.let("x", a)
        x2 = ib.let("x", a)  # Second assignment violates SSA

        ib.return_stmt(x2)

    func = f.get_result()
    program = ir.Program([func], "test_program", ir.Span.unknown())

    verify_pass = passes.verify_ssa()
    result_program = verify_pass(program)
    assert result_program is not None


def test_verify_ssa_name_shadowing():
    """Test VerifySSA detects name shadowing."""
    ib = builder.IRBuilder()

    with ib.function("test_shadow") as f:
        a = f.param("a", ir.ScalarType(DataType.INT64))
        f.return_type(ir.ScalarType(DataType.INT64))

        outer_i = ib.let("i", a)

        loop_var = ib.var("i", ir.ScalarType(DataType.INT64))  # Shadows outer 'i'
        with ib.for_loop(loop_var, 0, 5, 1):
            _tmp = ib.let("tmp", loop_var)

        ib.return_stmt(outer_i)

    func = f.get_result()
    program = ir.Program([func], "test_program", ir.Span.unknown())

    verify_pass = passes.verify_ssa()
    result_program = verify_pass(program)
    assert result_program is not None


def test_verify_ssa_missing_yield():
    """Test VerifySSA detects missing yield in ForStmt."""
    span = ir.Span.unknown()

    a = ir.Var("a", ir.ScalarType(DataType.INT64), span)
    params: list[ir.Var] = [a]
    return_types: list[ir.Type] = [ir.ScalarType(DataType.INT64)]

    loop_var = ir.Var("i", ir.ScalarType(DataType.INT64), span)
    iter_arg = ir.IterArg("sum", ir.ScalarType(DataType.INT64), a, span)
    body = ir.AssignStmt(ir.Var("dummy", ir.ScalarType(DataType.INT64), span), loop_var, span)  # No yield!
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
    func = ir.Function("test_missing_yield", params, return_types, func_body, span)
    program = ir.Program([func], "test_program", span)

    verify_pass = passes.verify_ssa()
    result_program = verify_pass(program)
    assert result_program is not None


def test_verify_ssa_missing_else():
    """Test VerifySSA detects missing else branch."""
    span = ir.Span.unknown()

    a = ir.Var("a", ir.ScalarType(DataType.INT64), span)
    params: list[ir.Var] = [a]
    return_types: list[ir.Type] = [ir.ScalarType(DataType.INT64)]

    condition = ir.Gt(a, ir.ConstInt(0, DataType.INT64, span), DataType.BOOL, span)
    then_body = ir.YieldStmt([a], span)
    result_var = ir.Var("result", ir.ScalarType(DataType.INT64), span)

    if_stmt = ir.IfStmt(condition, then_body, None, [result_var], span)  # Missing else

    func_body = ir.SeqStmts([if_stmt, ir.ReturnStmt([result_var], span)], span)
    func = ir.Function("test_missing_else", params, return_types, func_body, span)
    program = ir.Program([func], "test_program", span)

    verify_pass = passes.verify_ssa()
    result_program = verify_pass(program)
    assert result_program is not None


def test_verify_ssa_valid_control_flow():
    """Test valid control flow passes verification."""
    span = ir.Span.unknown()

    a = ir.Var("a", ir.ScalarType(DataType.INT64), span)
    params: list[ir.Var] = [a]
    return_types: list[ir.Type] = [ir.ScalarType(DataType.INT64)]

    # Valid ForStmt
    loop_var = ir.Var("i", ir.ScalarType(DataType.INT64), span)
    iter_arg = ir.IterArg("sum", ir.ScalarType(DataType.INT64), a, span)
    yield_value = ir.Add(iter_arg, loop_var, DataType.INT64, span)
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
    func = ir.Function("test_valid", params, return_types, func_body, span)
    program = ir.Program([func], "test_program", span)

    verify_pass = passes.verify_ssa()
    result_program = verify_pass(program)
    assert result_program is not None


if __name__ == "__main__":
    test_verify_ssa_valid()
    test_verify_ssa_multiple_assignment()
    test_verify_ssa_name_shadowing()
    test_verify_ssa_missing_yield()
    test_verify_ssa_missing_else()
    test_verify_ssa_valid_control_flow()
