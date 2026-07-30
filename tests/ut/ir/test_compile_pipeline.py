# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for the shared IR pass pipeline."""

import pytest
from pypto import DataType, ir
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


def test_run_pass_pipeline_preserves_outer_instruments():
    seen: list[str] = []
    instrument = passes.CallbackInstrument(
        before_pass=lambda pass_obj, _program: seen.append(pass_obj.get_name()),
        name="outer",
    )
    with passes.PassContext([instrument], verification_level=passes.VerificationLevel.NONE):
        result = _run_pass_pipeline(_scalar_program(), operation="lower")
    assert isinstance(result.transformed_program, ir.Program)
    assert seen


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


def test_compile_validates_pass_context_conflict_before_creating_output(tmp_path):
    output_dir = tmp_path / "must_not_exist"
    with passes.PassContext([]):
        with pytest.raises(RuntimeError, match=r"compile\(\).*diagnostic_phase"):
            ir.compile(
                _scalar_program(),
                output_dir=str(output_dir),
                diagnostic_phase=passes.DiagnosticPhase.POST_PASS,
                dump_passes=False,
                skip_ptoas=True,
            )
    assert not output_dir.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
