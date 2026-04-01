# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for the unused variable warning verifier."""

import pytest
from pypto import DataType, ir, passes
from pypto.ir import builder


def _run_unused_var_check(program: ir.Program) -> list[passes.Diagnostic]:
    """Run the UnusedVariable warning check and return all warnings."""
    all_checks = passes.WarningVerifierRegistry.get_all_checks()
    return passes.WarningVerifierRegistry.run_checks(all_checks, program)


def _warnings(diagnostics: list[passes.Diagnostic]) -> list[passes.Diagnostic]:
    """Filter to only Warning-severity diagnostics."""
    return [d for d in diagnostics if d.severity == passes.DiagnosticSeverity.Warning]


def _warning_names(diagnostics: list[passes.Diagnostic]) -> list[str]:
    """Extract variable names from warning messages."""
    result = []
    for d in _warnings(diagnostics):
        # Messages have format: "Unused variable 'name' in function 'func'"
        start = d.message.find("'") + 1
        end = d.message.find("'", start)
        if start > 0 and end > start:
            result.append(d.message[start:end])
    return result


# ---------------------------------------------------------------------------
# Cases that should NOT produce warnings
# ---------------------------------------------------------------------------


def test_no_warning_all_vars_used():
    """All variables are used — no warnings."""
    ib = builder.IRBuilder()

    with ib.function("all_used") as f:
        a = f.param("a", ir.ScalarType(DataType.INT64))
        f.return_type(ir.ScalarType(DataType.INT64))
        x = ib.let("x", a)
        ib.return_stmt(x)

    program = ir.Program([f.get_result()], "prog", ir.Span.unknown())
    assert len(_warnings(_run_unused_var_check(program))) == 0


def test_no_warning_function_param_unused():
    """Function parameters should NOT produce unused warnings (they're part of the interface)."""
    ib = builder.IRBuilder()

    with ib.function("param_unused") as f:
        _unused_param = f.param("unused_param", ir.ScalarType(DataType.INT64))
        f.return_type(ir.ScalarType(DataType.INT64))
        ib.return_stmt(ir.ConstInt(0, DataType.INT64, ir.Span.unknown()))

    program = ir.Program([f.get_result()], "prog", ir.Span.unknown())
    assert len(_warnings(_run_unused_var_check(program))) == 0


def test_no_warning_loop_var():
    """Loop variables (ForStmt::loop_var_) should NOT produce unused warnings."""
    ib = builder.IRBuilder()

    with ib.function("loop_var") as f:
        n = f.param("n", ir.ScalarType(DataType.INT64))
        f.return_type(ir.ScalarType(DataType.INT64))
        i = ib.var("i", ir.ScalarType(DataType.INT64))
        with ib.for_loop(i, 0, n, 1):
            # Loop body doesn't use i — but i is a loop_var, not an assignment
            _dummy = ib.let("dummy", n)
        ib.return_stmt(n)

    program = ir.Program([f.get_result()], "prog", ir.Span.unknown())
    # 'dummy' IS unused but 'i' should not be warned about
    names = _warning_names(_run_unused_var_check(program))
    assert "i" not in names


def test_no_warning_var_used_in_rhs():
    """Variable used on the RHS of an assignment should not be warned."""
    ib = builder.IRBuilder()

    with ib.function("used_in_rhs") as f:
        a = f.param("a", ir.ScalarType(DataType.INT64))
        f.return_type(ir.ScalarType(DataType.INT64))
        x = ib.let("x", a)
        y = ib.let("y", x)
        ib.return_stmt(y)

    program = ir.Program([f.get_result()], "prog", ir.Span.unknown())
    assert len(_warnings(_run_unused_var_check(program))) == 0


# ---------------------------------------------------------------------------
# Cases that SHOULD produce warnings
# ---------------------------------------------------------------------------


def test_warn_unused_assigned_var():
    """Variable assigned but never read should produce a warning."""
    ib = builder.IRBuilder()

    with ib.function("unused_assign") as f:
        a = f.param("a", ir.ScalarType(DataType.INT64))
        f.return_type(ir.ScalarType(DataType.INT64))
        _unused = ib.let("unused", ir.ConstInt(42, DataType.INT64, ir.Span.unknown()))
        ib.return_stmt(a)

    program = ir.Program([f.get_result()], "prog", ir.Span.unknown())
    warnings = _warnings(_run_unused_var_check(program))
    assert len(warnings) == 1
    assert "unused" in warnings[0].message
    assert warnings[0].rule_name == "UnusedVariableCheck"


def test_warn_multiple_unused():
    """Multiple unused variables should each produce a warning."""
    ib = builder.IRBuilder()

    with ib.function("multi_unused") as f:
        a = f.param("a", ir.ScalarType(DataType.INT64))
        f.return_type(ir.ScalarType(DataType.INT64))
        _u1 = ib.let("unused1", ir.ConstInt(1, DataType.INT64, ir.Span.unknown()))
        _u2 = ib.let("unused2", ir.ConstInt(2, DataType.INT64, ir.Span.unknown()))
        ib.return_stmt(a)

    program = ir.Program([f.get_result()], "prog", ir.Span.unknown())
    names = _warning_names(_run_unused_var_check(program))
    assert "unused1" in names
    assert "unused2" in names


def test_warn_severity_is_warning():
    """Unused variable diagnostics should have Warning severity, not Error."""
    ib = builder.IRBuilder()

    with ib.function("severity_check") as f:
        a = f.param("a", ir.ScalarType(DataType.INT64))
        f.return_type(ir.ScalarType(DataType.INT64))
        _unused = ib.let("unused", ir.ConstInt(42, DataType.INT64, ir.Span.unknown()))
        ib.return_stmt(a)

    program = ir.Program([f.get_result()], "prog", ir.Span.unknown())
    diags = _run_unused_var_check(program)
    for d in diags:
        assert d.severity == passes.DiagnosticSeverity.Warning


# ---------------------------------------------------------------------------
# Warning system infrastructure tests
# ---------------------------------------------------------------------------


def test_warning_check_set():
    """WarningCheckSet basic operations."""
    wcs = passes.WarningCheckSet()
    assert wcs.empty()

    wcs.insert(passes.WarningCheck.UnusedVariable)
    assert not wcs.empty()
    assert wcs.contains(passes.WarningCheck.UnusedVariable)

    wcs.remove(passes.WarningCheck.UnusedVariable)
    assert wcs.empty()
    assert not wcs.contains(passes.WarningCheck.UnusedVariable)


def test_warning_check_set_difference():
    """WarningCheckSet.difference removes specified checks."""
    all_checks = passes.WarningVerifierRegistry.get_all_checks()
    assert all_checks.contains(passes.WarningCheck.UnusedVariable)

    disabled = passes.WarningCheckSet()
    disabled.insert(passes.WarningCheck.UnusedVariable)
    effective = all_checks.difference(disabled)
    assert not effective.contains(passes.WarningCheck.UnusedVariable)


def test_warning_level_enum():
    """WarningLevel enum has expected values."""
    assert passes.WarningLevel.NONE != passes.WarningLevel.PRE_PIPELINE
    assert passes.WarningLevel.PRE_PIPELINE != passes.WarningLevel.POST_PASS
    assert passes.WarningLevel.POST_PASS != passes.WarningLevel.BOTH


def test_pass_context_warning_config():
    """PassContext accepts and returns warning configuration."""
    disabled = passes.WarningCheckSet()
    disabled.insert(passes.WarningCheck.UnusedVariable)

    ctx = passes.PassContext(
        [],
        warning_level=passes.WarningLevel.BOTH,
        disabled_warnings=disabled,
    )
    assert ctx.get_warning_level() == passes.WarningLevel.BOTH
    assert ctx.get_disabled_warnings().contains(passes.WarningCheck.UnusedVariable)


def test_pass_context_default_warning_config():
    """PassContext defaults to PrePipeline warning level with no disabled warnings."""
    ctx = passes.PassContext([])
    assert ctx.get_warning_level() == passes.WarningLevel.PRE_PIPELINE
    assert ctx.get_disabled_warnings().empty()


def test_pipeline_disabled_warnings_no_output():
    """PassPipeline with WarningLevel.NONE should not emit warnings."""
    ib = builder.IRBuilder()

    with ib.function("pipeline_test") as f:
        a = f.param("a", ir.ScalarType(DataType.INT64))
        f.return_type(ir.ScalarType(DataType.INT64))
        _unused = ib.let("unused", ir.ConstInt(42, DataType.INT64, ir.Span.unknown()))
        ib.return_stmt(a)

    program = ir.Program([f.get_result()], "prog", ir.Span.unknown())

    # Disable verification to avoid structural errors, disable warnings
    ctx = passes.PassContext(
        [],
        verification_level=passes.VerificationLevel.NONE,
        warning_level=passes.WarningLevel.NONE,
    )
    with ctx:
        pipeline = passes.PassPipeline()
        # Should not crash or emit warnings
        pipeline.run(program)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
