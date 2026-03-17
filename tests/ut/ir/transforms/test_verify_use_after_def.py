# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for UseAfterDef structural property verifier."""

import pytest
from pypto import DataType, ir, passes
from pypto.ir import builder


def _use_after_def_props() -> passes.IRPropertySet:
    props = passes.IRPropertySet()
    props.insert(passes.IRProperty.UseAfterDef)
    return props


def _errors(diagnostics: list[passes.Diagnostic]) -> list[passes.Diagnostic]:
    return [d for d in diagnostics if d.severity == passes.DiagnosticSeverity.Error]


# ---------------------------------------------------------------------------
# Valid cases
# ---------------------------------------------------------------------------


def test_valid_simple():
    """Variable defined (via param) before use — no error."""
    ib = builder.IRBuilder()

    with ib.function("valid_simple") as f:
        a = f.param("a", ir.ScalarType(DataType.INT64))
        f.return_type(ir.ScalarType(DataType.INT64))
        x = ib.let("x", a)
        ib.return_stmt(x)

    program = ir.Program([f.get_result()], "prog", ir.Span.unknown())
    assert len(_errors(passes.PropertyVerifierRegistry.verify(_use_after_def_props(), program))) == 0


def test_valid_sequential_assigns():
    """Chained assignments each reference previously defined variables."""
    ib = builder.IRBuilder()

    with ib.function("valid_chain") as f:
        a = f.param("a", ir.ScalarType(DataType.INT64))
        f.return_type(ir.ScalarType(DataType.INT64))
        x = ib.let("x", a)
        y = ib.let("y", x)
        z = ib.let("z", y)
        ib.return_stmt(z)

    program = ir.Program([f.get_result()], "prog", ir.Span.unknown())
    assert len(_errors(passes.PropertyVerifierRegistry.verify(_use_after_def_props(), program))) == 0


def test_valid_for_loop_var_in_body():
    """Loop variable is valid inside the loop body."""
    ib = builder.IRBuilder()

    with ib.function("valid_for") as f:
        n = f.param("n", ir.ScalarType(DataType.INT64))
        f.return_type(ir.ScalarType(DataType.INT64))
        i = ib.var("i", ir.ScalarType(DataType.INT64))
        with ib.for_loop(i, 0, n, 1):
            _tmp = ib.let("tmp", i)
        ib.return_stmt(n)

    program = ir.Program([f.get_result()], "prog", ir.Span.unknown())
    assert len(_errors(passes.PropertyVerifierRegistry.verify(_use_after_def_props(), program))) == 0


def test_valid_return_var_after_for():
    """return_var from a for loop is accessible after the loop ends."""
    span = ir.Span.unknown()
    a = ir.Var("a", ir.ScalarType(DataType.INT64), span)
    loop_var = ir.Var("i", ir.ScalarType(DataType.INT64), span)
    iter_arg = ir.IterArg("acc", ir.ScalarType(DataType.INT64), a, span)
    result_var = ir.Var("result", ir.ScalarType(DataType.INT64), span)

    yield_stmt = ir.YieldStmt([iter_arg], span)
    for_stmt = ir.ForStmt(
        loop_var,
        ir.ConstInt(0, DataType.INT64, span),
        ir.ConstInt(10, DataType.INT64, span),
        ir.ConstInt(1, DataType.INT64, span),
        [iter_arg],
        yield_stmt,
        [result_var],
        span,
    )
    # result_var is defined by the for loop — using it after is valid
    ret = ir.ReturnStmt([result_var], span)
    body = ir.SeqStmts([for_stmt, ret], span)
    func = ir.Function("valid_rv", [a], [ir.ScalarType(DataType.INT64)], body, span)
    program = ir.Program([func], "prog", span)

    assert len(_errors(passes.PropertyVerifierRegistry.verify(_use_after_def_props(), program))) == 0


# ---------------------------------------------------------------------------
# Invalid cases
# ---------------------------------------------------------------------------


def test_invalid_use_before_def():
    """Variable x used in RHS of an AssignStmt before x is defined."""
    span = ir.Span.unknown()
    a = ir.Var("a", ir.ScalarType(DataType.INT64), span)
    x = ir.Var("x", ir.ScalarType(DataType.INT64), span)
    y = ir.Var("y", ir.ScalarType(DataType.INT64), span)

    # use_x references x before def_x defines it
    use_x = ir.AssignStmt(y, x, span)
    def_x = ir.AssignStmt(x, a, span)
    ret = ir.ReturnStmt([a], span)

    body = ir.SeqStmts([use_x, def_x, ret], span)
    func = ir.Function("bad_func", [a], [ir.ScalarType(DataType.INT64)], body, span)
    program = ir.Program([func], "prog", span)

    errors = _errors(passes.PropertyVerifierRegistry.verify(_use_after_def_props(), program))
    assert len(errors) >= 1
    assert any("x" in d.message for d in errors)


def test_invalid_loop_var_used_after_loop():
    """Loop variable is out of scope outside the loop body."""
    span = ir.Span.unknown()
    a = ir.Var("a", ir.ScalarType(DataType.INT64), span)
    loop_var = ir.Var("i", ir.ScalarType(DataType.INT64), span)
    y = ir.Var("y", ir.ScalarType(DataType.INT64), span)

    for_body = ir.ReturnStmt([], span)
    for_stmt = ir.ForStmt(
        loop_var,
        ir.ConstInt(0, DataType.INT64, span),
        ir.ConstInt(10, DataType.INT64, span),
        ir.ConstInt(1, DataType.INT64, span),
        [],
        for_body,
        [],
        span,
    )
    # loop_var is used after the loop — it is no longer in scope
    use_after = ir.AssignStmt(y, loop_var, span)
    ret = ir.ReturnStmt([a], span)

    body = ir.SeqStmts([for_stmt, use_after, ret], span)
    func = ir.Function("loop_escape", [a], [ir.ScalarType(DataType.INT64)], body, span)
    program = ir.Program([func], "prog", span)

    errors = _errors(passes.PropertyVerifierRegistry.verify(_use_after_def_props(), program))
    assert len(errors) >= 1
    assert any("i" in d.message for d in errors)


def test_invalid_branch_def_not_visible_outside():
    """A variable defined only inside an if-branch must not be visible outside."""
    span = ir.Span.unknown()
    cond_var = ir.Var("cond", ir.ScalarType(DataType.BOOL), span)
    x = ir.Var("x", ir.ScalarType(DataType.INT64), span)
    y = ir.Var("y", ir.ScalarType(DataType.INT64), span)

    # x is defined only in then_body
    then_body = ir.AssignStmt(x, ir.ConstInt(1, DataType.INT64, span), span)
    if_stmt = ir.IfStmt(cond_var, then_body, None, [], span)

    # x is used after the if — but it may not have been defined (else branch missing)
    use_x = ir.AssignStmt(y, x, span)
    ret = ir.ReturnStmt([cond_var], span)

    body = ir.SeqStmts([if_stmt, use_x, ret], span)
    func = ir.Function("branch_escape", [cond_var], [ir.ScalarType(DataType.BOOL)], body, span)
    program = ir.Program([func], "prog", span)

    errors = _errors(passes.PropertyVerifierRegistry.verify(_use_after_def_props(), program))
    assert len(errors) >= 1
    assert any("x" in d.message for d in errors)


def test_error_code():
    """UseAfterDef errors carry error code 401."""
    span = ir.Span.unknown()
    a = ir.Var("a", ir.ScalarType(DataType.INT64), span)
    x = ir.Var("x", ir.ScalarType(DataType.INT64), span)
    y = ir.Var("y", ir.ScalarType(DataType.INT64), span)

    use_x = ir.AssignStmt(y, x, span)
    ret = ir.ReturnStmt([a], span)
    body = ir.SeqStmts([use_x, ret], span)
    func = ir.Function("err_code_func", [a], [ir.ScalarType(DataType.INT64)], body, span)
    program = ir.Program([func], "prog", span)

    errors = _errors(passes.PropertyVerifierRegistry.verify(_use_after_def_props(), program))
    assert len(errors) >= 1
    assert all(d.error_code == 401 for d in errors)
    assert all(d.rule_name == "UseAfterDefCheck" for d in errors)


def test_use_after_def_is_structural_property():
    """UseAfterDef must be present in GetStructuralProperties()."""
    structural = passes.get_structural_properties()
    assert structural.contains(passes.IRProperty.UseAfterDef)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
