# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for the unified Diagnostic system: registry, instrument, and file output."""

from __future__ import annotations

import re

import pypto.language as pl
import pytest
from pypto import DataType, backend, ir, passes
from pypto.backend import BackendType
from pypto.ir import builder


@pytest.fixture(autouse=True)
def reset_backend_around_test():
    backend.reset_for_testing()
    yield
    backend.reset_for_testing()


def _make_program_with_perf_hint(innermost: int = 16) -> ir.Program:
    """Build an InCore program whose tile.load triggers PH001 on Ascend950."""
    rows = 16

    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            x: pl.Tensor[[rows, innermost], pl.FP32],
            out: pl.Out[pl.Tensor[[rows, innermost], pl.FP32]],
        ) -> pl.Tensor[[rows, innermost], pl.FP32]:
            t: pl.Tile[[rows, innermost], pl.FP32] = pl.load(x, [0, 0], [rows, innermost])
            out_1: pl.Tensor[[rows, innermost], pl.FP32] = pl.store(t, [0, 0], out)
            return out_1

    return Prog


def _make_program_with_warning_and_perf_hint() -> ir.Program:
    """Build a program that emits both UnusedVariable and PH001 diagnostics."""

    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            x: pl.Tensor[[16, 16], pl.FP32],
            out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            unused: pl.Tile[[16, 16], pl.FP32] = pl.load(x, [0, 0], [16, 16])  # noqa: F841
            t: pl.Tile[[16, 16], pl.FP32] = pl.load(x, [0, 0], [16, 16])
            return pl.store(t, [0, 0], out)

    return Prog


def _collect_diagnostics(program: ir.Program, checks: passes.DiagnosticCheckSet) -> list:
    """Collect pre- and post-pipeline diagnostics without relying on stderr capture.

    Both phases are always queried; phase-specific check sets may return an empty
    list for the phase where no selected check is registered.
    """
    pre = passes.DiagnosticCheckRegistry.run_checks(checks, passes.DiagnosticPhase.PRE_PIPELINE, program)
    post = passes.DiagnosticCheckRegistry.run_checks(checks, passes.DiagnosticPhase.POST_PIPELINE, program)
    return list(pre) + list(post)


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------


def test_get_perf_hint_checks_returns_only_perf_hints():
    """get_perf_hint_checks excludes warnings."""
    perf = passes.DiagnosticCheckRegistry.get_perf_hint_checks()
    warns = passes.DiagnosticCheckRegistry.get_warning_checks()
    assert perf.contains(passes.DiagnosticCheck.TileInnermostDimGranularity)
    assert not perf.contains(passes.DiagnosticCheck.UnusedVariable)
    assert warns.contains(passes.DiagnosticCheck.UnusedVariable)
    assert not warns.contains(passes.DiagnosticCheck.TileInnermostDimGranularity)


def test_all_checks_is_union_of_warnings_and_perf_hints():
    """get_all_checks == warnings + perf hints."""
    perf = passes.DiagnosticCheckRegistry.get_perf_hint_checks()
    warns = passes.DiagnosticCheckRegistry.get_warning_checks()
    all_checks = passes.DiagnosticCheckRegistry.get_all_checks()
    assert all_checks == warns.union_with(perf)


def test_run_checks_filters_by_phase():
    """Running PrePipeline phase only runs Warning-severity checks (registered there)."""
    backend.set_backend_type(BackendType.Ascend950)
    program = _make_program_with_perf_hint(16)
    all_checks = passes.DiagnosticCheckRegistry.get_all_checks()
    pre = passes.DiagnosticCheckRegistry.run_checks(all_checks, passes.DiagnosticPhase.PRE_PIPELINE, program)
    post = passes.DiagnosticCheckRegistry.run_checks(
        all_checks, passes.DiagnosticPhase.POST_PIPELINE, program
    )
    pre_perf = [d for d in pre if d.severity == passes.DiagnosticSeverity.PerfHint]
    post_perf = [d for d in post if d.severity == passes.DiagnosticSeverity.PerfHint]
    assert pre_perf == []  # perf hints registered at POST_PIPELINE, not PRE
    assert len(post_perf) >= 1


# ---------------------------------------------------------------------------
# Diagnostic struct
# ---------------------------------------------------------------------------


def test_diagnostic_carries_hint_code():
    """A PerfHint diagnostic carries its hint_code (PH001) through the binding."""
    backend.set_backend_type(BackendType.Ascend950)
    program = _make_program_with_perf_hint(16)
    checks = passes.DiagnosticCheckSet()
    checks.insert(passes.DiagnosticCheck.TileInnermostDimGranularity)
    diags = passes.DiagnosticCheckRegistry.run_checks(checks, passes.DiagnosticPhase.POST_PIPELINE, program)
    perf = [d for d in diags if d.severity == passes.DiagnosticSeverity.PerfHint]
    assert len(perf) >= 1
    assert perf[0].hint_code == "PH001"


def test_warning_has_empty_hint_code():
    """Warning diagnostics carry an empty hint_code.

    Constructs a program with an unused variable so the UnusedVariable warning
    actually fires, then asserts the registry stamps an empty hint_code on the
    resulting Warning-severity diagnostic. Catches regressions where a warning
    accidentally inherits a perf-hint code.
    """
    ib = builder.IRBuilder()
    with ib.function("warn_no_hint_code") as f:
        a = f.param("a", ir.ScalarType(DataType.INT64))
        f.return_type(ir.ScalarType(DataType.INT64))
        _unused = ib.let("unused", ir.ConstInt(42, DataType.INT64, ir.Span.unknown()))
        ib.return_stmt(a)
    program = ir.Program([f.get_result()], "prog", ir.Span.unknown())

    checks = passes.DiagnosticCheckSet()
    checks.insert(passes.DiagnosticCheck.UnusedVariable)
    diags = passes.DiagnosticCheckRegistry.run_checks(checks, passes.DiagnosticPhase.PRE_PIPELINE, program)
    warns = [d for d in diags if d.severity == passes.DiagnosticSeverity.Warning]
    assert len(warns) >= 1
    for w in warns:
        assert w.hint_code == "", f"Warning should have empty hint_code, got {w.hint_code!r}"


# ---------------------------------------------------------------------------
# Suppression
# ---------------------------------------------------------------------------


def test_disabled_diagnostics_suppresses_check():
    """A check listed in disabled_diagnostics doesn't run via PassPipeline."""
    backend.set_backend_type(BackendType.Ascend950)
    program = _make_program_with_perf_hint(16)
    disabled = passes.DiagnosticCheckSet()
    disabled.insert(passes.DiagnosticCheck.UnusedControlFlowResult)
    disabled.insert(passes.DiagnosticCheck.TileInnermostDimGranularity)

    all_checks = passes.DiagnosticCheckRegistry.get_all_checks()
    effective = all_checks.difference(disabled)
    diags = passes.DiagnosticCheckRegistry.run_checks(
        effective, passes.DiagnosticPhase.POST_PIPELINE, program
    )
    perf = [d for d in diags if d.severity == passes.DiagnosticSeverity.PerfHint]
    assert perf == []


# ---------------------------------------------------------------------------
# File output via ReportInstrument.output_dir
# ---------------------------------------------------------------------------


def _run_pipeline_with_perf_hint(instruments, dphase=passes.DiagnosticPhase.PRE_PIPELINE):
    """Run an empty PassPipeline so PostPipeline diagnostics fire once.

    PassPipeline::Run unconditionally executes the PostPipeline phase as long
    as the diagnostic phase gate is not None — even with zero passes.
    """
    backend.set_backend_type(BackendType.Ascend950)
    program = _make_program_with_perf_hint(16)
    ctx = passes.PassContext(
        instruments,
        verification_level=passes.VerificationLevel.NONE,
        diagnostic_phase=dphase,
    )
    with ctx:
        pipeline = passes.PassPipeline()
        pipeline.run(program)
    return program


def test_perf_hint_log_file_appended_when_report_instrument_present(tmp_path):
    """With a ReportInstrument in the context, perf_hints.log is written."""
    report = passes.ReportInstrument(str(tmp_path))
    _run_pipeline_with_perf_hint([report])

    log = tmp_path / "perf_hints.log"
    assert log.exists()
    text = log.read_text()
    # One line per emitted hint, prefixed with [perf_hint PH001]
    lines = [line for line in text.splitlines() if line.strip()]
    assert len(lines) >= 1
    for entry in lines:
        assert entry.startswith("[perf_hint PH001]"), f"Unexpected line: {entry!r}"
        assert "TileInnermostDimGranularity" in entry


def test_perf_hint_log_file_not_created_without_report_instrument(tmp_path, monkeypatch):
    """Without a ReportInstrument, no file is written."""
    # Run in a scratch CWD so we'd notice any stray writes.
    monkeypatch.chdir(tmp_path)
    _run_pipeline_with_perf_hint([])
    assert not (tmp_path / "perf_hints.log").exists()


def test_perf_hint_log_file_appends_across_runs(tmp_path):
    """Two pipeline runs through the same ReportInstrument both write to the file."""
    report = passes.ReportInstrument(str(tmp_path))
    _run_pipeline_with_perf_hint([report])
    first = (tmp_path / "perf_hints.log").read_text().splitlines()
    _run_pipeline_with_perf_hint([report])
    second = (tmp_path / "perf_hints.log").read_text().splitlines()
    # File grows: second run appends without truncating.
    assert len(second) > len(first)


def test_warning_does_not_appear_in_perf_hints_log(tmp_path, capfd):
    """Only PerfHint-severity diagnostics flow to the file; warnings stay on stderr.

    Builds a program that emits both a Warning (UnusedVariable) and a PerfHint
    (TileInnermostDimGranularity), runs the pipeline through a ReportInstrument,
    and asserts `perf_hints.log` is created and contains only PerfHint lines.
    The diagnostics are collected directly first so the file assertion is not
    vacuous even on platforms where native stderr capture is unavailable.
    """

    backend.set_backend_type(BackendType.Ascend950)

    program = _make_program_with_warning_and_perf_hint()

    enabled_checks = passes.DiagnosticCheckSet()
    enabled_checks.insert(passes.DiagnosticCheck.UnusedVariable)
    enabled_checks.insert(passes.DiagnosticCheck.TileInnermostDimGranularity)
    diags = _collect_diagnostics(program, enabled_checks)
    assert any(d.rule_name == "UnusedVariableCheck" for d in diags)
    assert any(d.hint_code == "PH001" for d in diags)

    report = passes.ReportInstrument(str(tmp_path))
    ctx = passes.PassContext(
        [report],
        verification_level=passes.VerificationLevel.NONE,
        diagnostic_phase=passes.DiagnosticPhase.PRE_PIPELINE,
        disabled_diagnostics=passes.DiagnosticCheckSet(),  # warning enabled
    )
    with ctx:
        passes.PassPipeline().run(program)

    captured = capfd.readouterr()
    combined = captured.out + captured.err

    log = tmp_path / "perf_hints.log"
    assert log.exists(), "perf_hints.log was not created"
    text = log.read_text()
    assert "[perf_hint PH001]" in text
    # Native stderr capture is platform-dependent on the macOS runner, but when
    # it is captured the warning should route to stderr instead of the file.
    if combined:
        assert "UnusedVariableCheck" in combined
    assert "[warning]" not in text


# ---------------------------------------------------------------------------
# stderr surfacing
# ---------------------------------------------------------------------------


def test_perf_hint_visible_at_default_log_level(capfd):
    """The default diagnostic path emits PH001 and checks stderr when capture is present."""
    backend.set_backend_type(BackendType.Ascend950)
    program = _make_program_with_perf_hint(16)
    checks = passes.DiagnosticCheckSet()
    checks.insert(passes.DiagnosticCheck.TileInnermostDimGranularity)
    diags = passes.DiagnosticCheckRegistry.run_checks(checks, passes.DiagnosticPhase.POST_PIPELINE, program)
    assert any(d.hint_code == "PH001" for d in diags)

    ctx = passes.PassContext(
        [],
        verification_level=passes.VerificationLevel.NONE,
        diagnostic_phase=passes.DiagnosticPhase.PRE_PIPELINE,
    )
    with ctx:
        pipeline = passes.PassPipeline()
        pipeline.run(program)
    captured = capfd.readouterr()
    combined = captured.out + captured.err
    if combined:
        assert re.search(r"\[perf_hint PH001\]", combined), f"perf hint not in output:\n{combined}"


# ---------------------------------------------------------------------------
# DiagnosticCheckSet is unhashable (mutable via insert/remove)
# ---------------------------------------------------------------------------


def test_diagnostic_check_set_hash_raises_typeerror():
    s = passes.DiagnosticCheckSet()
    with pytest.raises(TypeError, match="unhashable"):
        hash(s)


def test_diagnostic_check_set_use_as_set_member_raises():
    s = passes.DiagnosticCheckSet()
    with pytest.raises(TypeError, match="unhashable"):
        _ = {s}


def test_diagnostic_check_set_use_as_dict_key_raises():
    s = passes.DiagnosticCheckSet()
    with pytest.raises(TypeError, match="unhashable"):
        _ = {s: "value"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
