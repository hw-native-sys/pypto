# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for the shared IR pass pipeline."""

import json

import pytest
from pypto import DataType, ir
from pypto.compile_profiling import CompileProfiler
from pypto.ir.compile import _run_pass_pipeline
from pypto.pypto_core import passes


def _scalar_program() -> ir.Program:
    span = ir.Span.unknown()
    dtype = ir.ScalarType(DataType.INT64)
    x = ir.Var("x", dtype, span)
    y = ir.Var("y", dtype, span)
    body = ir.SeqStmts([ir.AssignStmt(y, x, span), ir.ReturnStmt([y], span)], span)
    fn = ir.Function("main", [x], [dtype], body, span)
    return ir.Program([fn], "lower_test", span)


def test_run_pass_pipeline_orders_outer_before_extra_instruments():
    seen: list[tuple[str, str]] = []
    outer_instrument = passes.CallbackInstrument(
        before_pass=lambda pass_obj, _program: seen.append(("outer", pass_obj.get_name())),
        name="outer",
    )
    extra_instrument = passes.CallbackInstrument(
        before_pass=lambda pass_obj, _program: seen.append(("extra", pass_obj.get_name())),
        name="extra",
    )
    with passes.PassContext([outer_instrument], verification_level=passes.VerificationLevel.NONE):
        result = _run_pass_pipeline(
            _scalar_program(),
            operation="lower",
            extra_instruments=(extra_instrument,),
        )
    assert isinstance(result.transformed_program, ir.Program)
    assert seen
    assert len(seen) % 2 == 0
    for outer_event, extra_event in zip(seen[::2], seen[1::2], strict=True):
        assert outer_event[0] == "outer"
        assert extra_event == ("extra", outer_event[1])


def test_run_pass_pipeline_names_diagnostic_conflict_for_lower():
    with passes.PassContext([]):
        with pytest.raises(RuntimeError, match=r"lower\(\).*diagnostic_phase"):
            _run_pass_pipeline(
                _scalar_program(),
                operation="lower",
                diagnostic_phase=passes.DiagnosticPhase.POST_PASS,
            )


def test_compile_validates_platform_before_creating_output(tmp_path):
    output_dir = tmp_path / "must_not_exist"
    with pytest.raises(ValueError, match="Invalid platform"):
        ir.compile(
            _scalar_program(),
            output_dir=str(output_dir),
            platform="invalid",
            dump_passes=False,
            skip_ptoas=True,
        )
    assert not output_dir.exists()


def test_compile_creates_output_before_pass_context_conflict(tmp_path):
    output_dir = tmp_path / "compile_output"
    with passes.PassContext([]):
        with pytest.raises(RuntimeError, match=r"compile\(\).*diagnostic_phase"):
            ir.compile(
                _scalar_program(),
                output_dir=str(output_dir),
                diagnostic_phase=passes.DiagnosticPhase.POST_PASS,
                dump_passes=False,
                skip_ptoas=True,
            )
    assert output_dir.is_dir()
    assert not (output_dir / "report").exists()


def test_compile_preserves_dump_and_report_artifacts(tmp_path):
    output_dir = tmp_path / "compile_output"
    ir.compile(
        _scalar_program(),
        output_dir=str(output_dir),
        dump_passes=True,
        skip_ptoas=True,
    )

    passes_dump = output_dir / "passes_dump"
    assert (passes_dump / "00_frontend.py").is_file()
    assert list(passes_dump.glob("*_after_*.py"))
    assert (output_dir / "report").is_dir()


def test_compile_owned_profiler_writes_nested_pass_and_codegen_stages(tmp_path):
    output_dir = tmp_path / "owned_profile"
    assert CompileProfiler.current() is None
    ir.compile(
        _scalar_program(),
        output_dir=str(output_dir),
        dump_passes=False,
        skip_ptoas=True,
        profiling=True,
    )
    assert CompileProfiler.current() is None

    report_path = output_dir / "report" / "pipeline_profile.json"
    profile = json.loads(report_path.read_text())
    stages = profile["stages"]
    assert [stage["name"] for stage in stages] == ["passes", "codegen"]
    assert stages[0]["children"]


def test_compile_outer_profiler_retains_ownership(tmp_path):
    output_dir = tmp_path / "outer_profile"
    with CompileProfiler() as profiler:
        ir.compile(
            _scalar_program(),
            output_dir=str(output_dir),
            dump_passes=False,
            skip_ptoas=True,
            profiling=True,
        )
        assert CompileProfiler.current() is profiler
    assert CompileProfiler.current() is None

    stages = profiler.to_dict()["stages"]
    assert [stage["name"] for stage in stages] == ["passes", "codegen"]
    assert stages[0]["children"]
    assert not (output_dir / "report" / "pipeline_profile.json").exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
