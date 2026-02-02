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
from pypto import ir
from pypto.ir import builder
from pypto.pypto_core import DataType, passes


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
    ib = builder.IRBuilder()

    with ib.function("test_ssa_error") as f:
        a = f.param("a", ir.ScalarType(DataType.INT64))
        f.return_type(ir.ScalarType(DataType.INT64))

        _x = ib.let("x", a)
        x2 = ib.let("x", a)  # Multiple assignment

        ib.return_stmt(x2)

    func = f.get_result()
    program = ir.Program([func], "test_program", ir.Span.unknown())

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
    ib = builder.IRBuilder()

    with ib.function("test_disable") as f:
        a = f.param("a", ir.ScalarType(DataType.INT64))
        f.return_type(ir.ScalarType(DataType.INT64))

        _x = ib.let("x", a)
        x2 = ib.let("x", a)  # Multiple assignment

        ib.return_stmt(x2)

    func = f.get_result()
    program = ir.Program([func], "test_program", ir.Span.unknown())

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
    ib = builder.IRBuilder()

    with ib.function("test_throw") as f:
        a = f.param("a", ir.ScalarType(DataType.INT64))
        f.return_type(ir.ScalarType(DataType.INT64))

        _x = ib.let("x", a)
        x2 = ib.let("x", a)  # Multiple assignment

        ib.return_stmt(x2)

    func = f.get_result()
    program = ir.Program([func], "test_program", ir.Span.unknown())

    verifier = passes.IRVerifier.create_default()

    # Should raise VerificationError
    with pytest.raises(Exception):  # The exact exception type from C++
        verifier.verify_or_throw(program)


def test_verifier_generate_report():
    """Test generating verification report."""
    ib = builder.IRBuilder()

    with ib.function("test_report") as f:
        a = f.param("a", ir.ScalarType(DataType.INT64))
        f.return_type(ir.ScalarType(DataType.INT64))

        _x = ib.let("x", a)
        x2 = ib.let("x", a)  # Multiple assignment

        ib.return_stmt(x2)

    func = f.get_result()
    program = ir.Program([func], "test_program", ir.Span.unknown())

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
    ib = builder.IRBuilder()

    with ib.function("test_disabled") as f:
        a = f.param("a", ir.ScalarType(DataType.INT64))
        f.return_type(ir.ScalarType(DataType.INT64))

        _x = ib.let("x", a)
        x2 = ib.let("x", a)  # Multiple assignment

        ib.return_stmt(x2)

    func = f.get_result()
    program = ir.Program([func], "test_program", ir.Span.unknown())

    # Create verifier pass with SSA disabled
    verify_pass = passes.run_verifier(disabled_rules=["SSAVerify"])
    result_program = verify_pass(program)

    # Should still return the program (no exception)
    assert result_program is not None


def test_diagnostic_fields():
    """Test accessing Diagnostic fields."""
    ib = builder.IRBuilder()

    with ib.function("test_fields") as f:
        a = f.param("a", ir.ScalarType(DataType.INT64))
        f.return_type(ir.ScalarType(DataType.INT64))

        _x = ib.let("x", a)
        x2 = ib.let("x", a)

        ib.return_stmt(x2)

    func = f.get_result()
    program = ir.Program([func], "test_program", ir.Span.unknown())

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
