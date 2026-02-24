# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for IRVerifier."""

import pytest
from pypto import DataType, ir, passes
from pypto.ir import builder


def _make_ssa_violating_program() -> ir.Program:
    """Create a program with a genuine SSA violation (same Var assigned twice)."""
    span = ir.Span.unknown()
    scalar_type = ir.ScalarType(DataType.INT64)
    a = ir.Var("a", scalar_type, span)
    x = ir.Var("x", scalar_type, span)

    # Assign to the same Var pointer twice — genuine SSA violation
    assign1 = ir.AssignStmt(x, a, span)
    assign2 = ir.AssignStmt(x, a, span)
    return_stmt = ir.ReturnStmt([x], span)
    body = ir.SeqStmts([assign1, assign2, return_stmt], span)

    func = ir.Function("test_ssa_error", [a], [scalar_type], body, span)
    return ir.Program([func], "test_program", span)


def test_verifier_create_default():
    """Test creating default verifier."""
    verifier = passes.IRVerifier.create_default()
    assert verifier is not None


def test_verifier_empty():
    """Test creating empty verifier."""
    verifier = passes.IRVerifier()
    assert verifier is not None


def test_verifier_valid_program():
    """Test verifier on valid SSA program."""
    ib = builder.IRBuilder()

    with ib.function("test_valid") as f:
        a = f.param("a", ir.ScalarType(DataType.INT64))
        b = f.param("b", ir.ScalarType(DataType.INT64))
        f.return_type(ir.ScalarType(DataType.INT64))

        x = ib.let("x", a)
        ib.let("y", b)
        z = ib.let("z", x)

        ib.return_stmt(z)

    func = f.get_result()
    program = ir.Program([func], "test_program", ir.Span.unknown())

    # Create verifier and run verification
    verifier = passes.IRVerifier.create_default()
    diagnostics = verifier.verify(program)

    # Should have no diagnostics
    assert len(diagnostics) == 0


def test_verifier_ssa_error():
    """Test verifier detects SSA errors."""
    program = _make_ssa_violating_program()

    verifier = passes.IRVerifier.create_default()
    diagnostics = verifier.verify(program)

    # Should have at least one error
    assert len(diagnostics) > 0
    # All should be errors (not warnings)
    assert all(d.severity == passes.DiagnosticSeverity.Error for d in diagnostics)
    # At least one should be from SSAVerify rule
    assert any(d.rule_name == "SSAVerify" for d in diagnostics)


def test_verifier_disable_rule():
    """Test disabling verification rules."""
    program = _make_ssa_violating_program()

    # Create verifier and disable SSA verification
    verifier = passes.IRVerifier.create_default()
    verifier.disable_rule("SSAVerify")

    diagnostics = verifier.verify(program)

    # Should have no SSA errors since the rule is disabled
    assert all(d.rule_name != "SSAVerify" for d in diagnostics)


def test_verifier_enable_rule():
    """Test enabling a disabled rule."""
    verifier = passes.IRVerifier.create_default()

    # Disable and then re-enable
    verifier.disable_rule("SSAVerify")
    assert not verifier.is_rule_enabled("SSAVerify")

    verifier.enable_rule("SSAVerify")
    assert verifier.is_rule_enabled("SSAVerify")


def test_verifier_or_throw_no_error():
    """Test verify_or_throw on valid program (should not throw)."""
    ib = builder.IRBuilder()

    with ib.function("test_no_throw") as f:
        a = f.param("a", ir.ScalarType(DataType.INT64))
        f.return_type(ir.ScalarType(DataType.INT64))

        x = ib.let("x", a)
        ib.return_stmt(x)

    func = f.get_result()
    program = ir.Program([func], "test_program", ir.Span.unknown())

    verifier = passes.IRVerifier.create_default()
    # Should not raise exception
    verifier.verify_or_throw(program)


def test_verifier_or_throw_with_error():
    """Test verify_or_throw on invalid program (should throw)."""
    program = _make_ssa_violating_program()

    verifier = passes.IRVerifier.create_default()

    # Should raise VerificationError
    with pytest.raises(Exception):  # The exact exception type from C++
        verifier.verify_or_throw(program)


def test_verifier_generate_report():
    """Test generating verification report."""
    program = _make_ssa_violating_program()

    verifier = passes.IRVerifier.create_default()
    diagnostics = verifier.verify(program)

    # Generate report
    report = passes.IRVerifier.generate_report(diagnostics)

    # Report should contain key information
    assert "IR Verification Report" in report
    assert "SSAVerify" in report
    assert len(report) > 0


def test_verifier_as_pass():
    """Test using verifier as a Pass."""
    ib = builder.IRBuilder()

    with ib.function("test_pass") as f:
        a = f.param("a", ir.ScalarType(DataType.INT64))
        f.return_type(ir.ScalarType(DataType.INT64))

        x = ib.let("x", a)
        ib.return_stmt(x)

    func = f.get_result()
    program = ir.Program([func], "test_program", ir.Span.unknown())

    # Create verifier pass
    verify_pass = passes.run_verifier()
    result_program = verify_pass(program)

    # Should return the same program
    assert result_program is not None


def test_verifier_pass_with_disabled_rules():
    """Test verifier pass with disabled rules."""
    program = _make_ssa_violating_program()

    # Create verifier pass with SSA disabled
    verify_pass = passes.run_verifier(disabled_rules=["SSAVerify"])
    result_program = verify_pass(program)

    # Should still return the program (no exception)
    assert result_program is not None


def test_diagnostic_fields():
    """Test accessing Diagnostic fields."""
    program = _make_ssa_violating_program()

    verifier = passes.IRVerifier.create_default()
    diagnostics = verifier.verify(program)

    assert len(diagnostics) > 0

    # Check diagnostic fields
    diag = diagnostics[0]
    assert diag.severity in [passes.DiagnosticSeverity.Error, passes.DiagnosticSeverity.Warning]
    assert isinstance(diag.rule_name, str)
    assert isinstance(diag.error_code, int)
    assert isinstance(diag.message, str)
    assert diag.span is not None


def test_verifier_if_condition_scalar_type_invalid():
    """Test TypeCheck detects non-scalar IfStmt condition."""
    # Create a tensor type for condition (invalid)
    tensor_type = ir.TensorType([ir.ConstInt(4, DataType.INT64, ir.Span.unknown())], DataType.INT64)
    scalar_type = ir.ScalarType(DataType.INT64)

    # Create variables
    cond_var = ir.Var("cond", tensor_type, ir.Span.unknown())
    a_var = ir.Var("a", scalar_type, ir.Span.unknown())
    b_var = ir.Var("b", scalar_type, ir.Span.unknown())
    x_var = ir.Var("x", scalar_type, ir.Span.unknown())
    y_var = ir.Var("y", scalar_type, ir.Span.unknown())
    return_var = ir.Var("return_var", scalar_type, ir.Span.unknown())

    # Create statements
    assign_cond = ir.AssignStmt(cond_var, a_var, ir.Span.unknown())
    assign_x = ir.AssignStmt(x_var, b_var, ir.Span.unknown())
    assign_y = ir.AssignStmt(y_var, a_var, ir.Span.unknown())
    yield_then = ir.YieldStmt([x_var], ir.Span.unknown())
    yield_else = ir.YieldStmt([y_var], ir.Span.unknown())

    then_body = ir.SeqStmts([assign_x, yield_then], ir.Span.unknown())
    else_body = ir.SeqStmts([assign_y, yield_else], ir.Span.unknown())

    # Create IfStmt with TensorType condition
    if_stmt = ir.IfStmt(cond_var, then_body, else_body, [return_var], ir.Span.unknown())

    return_stmt = ir.ReturnStmt([return_var], ir.Span.unknown())
    body = ir.SeqStmts([assign_cond, if_stmt, return_stmt], ir.Span.unknown())

    func = ir.Function("test_if_invalid", [a_var, b_var], [scalar_type], body, ir.Span.unknown())
    program = ir.Program([func], "test_program", ir.Span.unknown())

    verifier = passes.IRVerifier.create_default()
    diagnostics = verifier.verify(program)

    # Should have TypeCheck error for condition with error code 106
    typecheck_diags = [d for d in diagnostics if d.rule_name == "TypeCheck" and d.error_code == 106]
    assert len(typecheck_diags) > 0
    assert "condition" in typecheck_diags[0].message.lower()
    assert "scalar" in typecheck_diags[0].message.lower()


def test_verifier_for_range_scalar_type_invalid():
    """Test TypeCheck detects non-scalar ForStmt range."""
    # Create types
    tensor_type = ir.TensorType([ir.ConstInt(4, DataType.INT64, ir.Span.unknown())], DataType.INT64)
    scalar_type = ir.ScalarType(DataType.INT64)

    # Create variables
    n_var = ir.Var("n", scalar_type, ir.Span.unknown())
    start_var = ir.Var("start", tensor_type, ir.Span.unknown())  # Invalid: TensorType
    stop_var = ir.Var("stop", tensor_type, ir.Span.unknown())  # Invalid: TensorType
    step_var = ir.Var("step", tensor_type, ir.Span.unknown())  # Invalid: TensorType
    i_var = ir.Var("i", scalar_type, ir.Span.unknown())
    sum_var = ir.Var("sum", scalar_type, ir.Span.unknown())
    iter_arg = ir.IterArg("iter_sum", scalar_type, sum_var, ir.Span.unknown())
    new_sum_var = ir.Var("new_sum", scalar_type, ir.Span.unknown())
    result_var = ir.Var("result", scalar_type, ir.Span.unknown())

    # Create statements
    assign_start = ir.AssignStmt(
        start_var, ir.ConstInt(0, DataType.INT64, ir.Span.unknown()), ir.Span.unknown()
    )
    assign_stop = ir.AssignStmt(stop_var, n_var, ir.Span.unknown())
    assign_step = ir.AssignStmt(
        step_var, ir.ConstInt(1, DataType.INT64, ir.Span.unknown()), ir.Span.unknown()
    )
    assign_sum = ir.AssignStmt(sum_var, ir.ConstInt(0, DataType.INT64, ir.Span.unknown()), ir.Span.unknown())

    assign_new_sum = ir.AssignStmt(new_sum_var, iter_arg, ir.Span.unknown())
    yield_stmt = ir.YieldStmt([new_sum_var], ir.Span.unknown())
    loop_body = ir.SeqStmts([assign_new_sum, yield_stmt], ir.Span.unknown())

    # Create ForStmt with TensorType range
    for_stmt = ir.ForStmt(
        i_var, start_var, stop_var, step_var, [iter_arg], loop_body, [result_var], ir.Span.unknown()
    )

    return_stmt = ir.ReturnStmt([result_var], ir.Span.unknown())
    body = ir.SeqStmts(
        [assign_start, assign_stop, assign_step, assign_sum, for_stmt, return_stmt], ir.Span.unknown()
    )

    func = ir.Function("test_for_invalid", [n_var], [scalar_type], body, ir.Span.unknown())
    program = ir.Program([func], "test_program", ir.Span.unknown())

    verifier = passes.IRVerifier.create_default()
    diagnostics = verifier.verify(program)

    # Should have TypeCheck errors for range (start, stop, step) with error code 107
    typecheck_diags = [d for d in diagnostics if d.rule_name == "TypeCheck" and d.error_code == 107]
    assert len(typecheck_diags) >= 3  # At least one for each: start, stop, step
    # Check that error messages mention range-related terms
    for diag in typecheck_diags:
        assert any(keyword in diag.message.lower() for keyword in ["start", "stop", "step"])
        assert "scalar" in diag.message.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
