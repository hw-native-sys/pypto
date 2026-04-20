# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for the ``pl.pipeline(..., stage=F)`` DSL iterator.

Validates the DSL surface and that the parser emits a ``ForStmt`` with
``kind == ForKind.Pipeline`` and ``attrs["pipeline_stages"] == F``. The actual
lowering is covered by ``test_lower_pipeline_loops.py``.
"""

from typing import cast

import pypto.language as pl
import pytest
from pypto import ir


def _outer_for(program: ir.Program) -> ir.ForStmt:
    func = list(program.functions.values())[0]
    body = cast(ir.SeqStmts, func.body)
    for stmt in body.stmts:
        if isinstance(stmt, ir.ForStmt):
            return stmt
    raise AssertionError("no ForStmt in function body")


class TestPipelineKwargParser:
    """Verify the parser emits kind=Pipeline + attrs[pipeline_stages]=F."""

    def test_pipeline_kind_and_stage_attr(self):
        @pl.program
        class P:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                for i in pl.pipeline(8, stage=4):
                    x = pl.add(x, 1.0)
                return x

        fs = _outer_for(P)
        assert fs.kind == ir.ForKind.Pipeline
        assert dict(fs.attrs).get("pipeline_stages") == 4

    def test_stage_one_is_accepted(self):
        @pl.program
        class P:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                for i in pl.pipeline(8, stage=1):
                    x = pl.add(x, 1.0)
                return x

        fs = _outer_for(P)
        assert fs.kind == ir.ForKind.Pipeline
        assert dict(fs.attrs).get("pipeline_stages") == 1

    def test_plain_range_has_no_pipeline_stages_attr(self):
        @pl.program
        class P:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                for i in pl.range(8):
                    x = pl.add(x, 1.0)
                return x

        fs = _outer_for(P)
        assert fs.kind == ir.ForKind.Sequential
        assert "pipeline_stages" not in dict(fs.attrs)

    def test_pipeline_with_init_values_parses(self):
        """``pl.pipeline`` composes with ``init_values`` — loop-carried state is supported."""

        @pl.program
        class P:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                for i, (acc,) in pl.pipeline(8, stage=2, init_values=(x,)):
                    acc = pl.add(acc, 1.0)
                    acc_rv = pl.yield_(acc)
                return acc_rv

        fs = _outer_for(P)
        assert fs.kind == ir.ForKind.Pipeline
        assert dict(fs.attrs).get("pipeline_stages") == 2
        assert len(fs.iter_args) == 1

    def test_round_trip_emits_pl_pipeline(self):
        """Printer surfaces stage= as kwarg; attr stripped from printed attrs dict."""

        @pl.program
        class P:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                for i in pl.pipeline(8, stage=4):
                    x = pl.add(x, 1.0)
                return x

        text = ir.python_print(P)
        assert "pl.pipeline(8, stage=4)" in text
        assert "pipeline_stages" not in text  # must not leak storage key
        # Round-trip via parse_program
        P_rt = pl.parse_program(text)
        assert ir.structural_equal(P, P_rt)


class TestPipelineKwargRejection:
    """The parser must reject invalid pl.pipeline calls with clear errors."""

    def test_missing_stage_rejected(self):
        with pytest.raises(Exception, match=r"pl\.pipeline\(\) requires stage="):

            @pl.program
            class _P:
                @pl.function
                def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                    for i in pl.pipeline(8):  # type: ignore[call-overload]
                        x = pl.add(x, 1.0)
                    return x

    def test_stage_zero_rejected(self):
        with pytest.raises(Exception, match=r"pl\.pipeline\(\) stage must be >= 1"):

            @pl.program
            class _P:
                @pl.function
                def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                    for i in pl.pipeline(8, stage=0):
                        x = pl.add(x, 1.0)
                    return x

    def test_stage_negative_rejected(self):
        with pytest.raises(Exception, match=r"pl\.pipeline\(\) stage must be >= 1"):

            @pl.program
            class _P:
                @pl.function
                def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                    for i in pl.pipeline(8, stage=-2):
                        x = pl.add(x, 1.0)
                    return x

    def test_stage_runtime_value_rejected(self):
        """Non-constant ``stage`` is rejected by the parser (must be a literal int)."""
        with pytest.raises(Exception, match=r"stage must be a compile-time constant"):

            @pl.program
            class _P:
                @pl.function
                def main(
                    self, x: pl.Tensor[[64], pl.FP32], n: pl.Scalar[pl.INT64]
                ) -> pl.Tensor[[64], pl.FP32]:
                    for i in pl.pipeline(8, stage=n):  # type: ignore[arg-type]
                        x = pl.add(x, 1.0)
                    return x

    def test_stage_on_range_rejected(self):
        with pytest.raises(Exception, match=r"stage= is only supported on pl\.pipeline"):

            @pl.program
            class _P:
                @pl.function
                def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                    for i in pl.range(8, stage=2):  # type: ignore[call-overload]
                        x = pl.add(x, 1.0)
                    return x

    def test_stage_on_unroll_rejected(self):
        with pytest.raises(Exception, match=r"stage= is only supported on pl\.pipeline"):

            @pl.program
            class _P:
                @pl.function
                def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                    for i in pl.unroll(8, stage=2):  # type: ignore[call-overload]
                        x = pl.add(x, 1.0)
                    return x

    def test_unroll_kwarg_on_range_is_gone(self):
        """``pl.range(..., unroll=)`` was removed in favor of pl.pipeline."""
        with pytest.raises(TypeError):
            pl.range(8, unroll=4)  # type: ignore[call-overload]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
