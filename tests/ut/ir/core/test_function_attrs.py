# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for Function-level attributes (attrs) system."""

import pypto.language as pl
import pypto.pypto_core as pc
import pytest
from pypto.pypto_core import ir


def _make_nop_body() -> ir.Stmt:
    """Create a minimal function body (pass statement)."""
    span = ir.Span.unknown()
    return ir.EvalStmt(ir.ConstInt(0, pc.DataType.INT64, span), span)


class TestFunctionAttrsBinding:
    """Test Function attrs via Python bindings."""

    def test_empty_attrs(self):
        span = ir.Span.unknown()
        body = _make_nop_body()
        func = ir.Function("f", [], [], body, span)
        assert func.attrs == {}
        assert func.split is None

    def test_split_via_attrs_param(self):
        span = ir.Span.unknown()
        body = _make_nop_body()
        func = ir.Function("f", [], [], body, span, attrs={"split": 1})
        assert func.attrs == {"split": 1}
        assert func.split == ir.SplitMode.UP_DOWN

    def test_left_right_via_attrs(self):
        span = ir.Span.unknown()
        body = _make_nop_body()
        func = ir.Function("f", [], [], body, span, attrs={"split": 2})
        assert func.attrs == {"split": 2}
        assert func.split == ir.SplitMode.LEFT_RIGHT

    def test_custom_attrs(self):
        span = ir.Span.unknown()
        body = _make_nop_body()
        func = ir.Function("f", [], [], body, span, attrs={"split": 1, "pipeline_depth": 2})
        assert func.attrs["split"] == 1
        assert func.attrs["pipeline_depth"] == 2
        assert func.split == ir.SplitMode.UP_DOWN

    def test_no_attrs_means_no_split(self):
        span = ir.Span.unknown()
        body = _make_nop_body()
        func = ir.Function("f", [], [], body, span)
        assert func.attrs == {}
        assert func.split is None


class TestFunctionAttrsDSL:
    """Test Function attrs via the @pl.function / @pl.program DSL decorators."""

    def test_program_split_up_down(self):
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.AIC, attrs={"split": pl.SplitMode.UP_DOWN})
            def main(self, x: pl.Tensor[[16, 128], pl.FP16]) -> pl.Tensor[[16, 128], pl.FP16]:
                return x

        func = Prog["main"]
        assert func is not None
        assert func.split == ir.SplitMode.UP_DOWN
        assert func.attrs["split"] == 1

    def test_program_no_attrs(self):
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.AIC)
            def main(self, x: pl.Tensor[[16, 128], pl.FP16]) -> pl.Tensor[[16, 128], pl.FP16]:
                return x

        func = Prog["main"]
        assert func is not None
        assert func.split is None
        assert func.attrs == {}

    def test_standalone_function_attrs(self):
        @pl.function(type=pl.FunctionType.AIV, attrs={"split": pl.SplitMode.LEFT_RIGHT})
        def my_func(x: pl.Tensor[[16, 128], pl.FP16]) -> pl.Tensor[[16, 128], pl.FP16]:
            return x

        assert my_func.split == ir.SplitMode.LEFT_RIGHT
        assert my_func.attrs["split"] == 2


class TestFunctionAttrsPrinterRoundtrip:
    """Test that Function attrs survive print -> parse round-trip."""

    @pytest.mark.skip(reason="pl.program_from_string not yet implemented")
    def test_split_roundtrip(self):
        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.AIC, attrs={"split": pl.SplitMode.UP_DOWN})
            def main(self, x: pl.Tensor[[16, 128], pl.FP16]) -> pl.Tensor[[16, 128], pl.FP16]:
                return x

        printed = ir.python_print(Before)
        After = pl.program_from_string(printed)  # type: ignore[attr-defined]
        func = After["main"]
        assert func.split == ir.SplitMode.UP_DOWN

    def test_structural_equality_with_attrs(self):
        @pl.program
        class A:
            @pl.function(type=pl.FunctionType.AIC, attrs={"split": pl.SplitMode.UP_DOWN})
            def main(self, x: pl.Tensor[[16, 128], pl.FP16]) -> pl.Tensor[[16, 128], pl.FP16]:
                return x

        @pl.program
        class B:
            @pl.function(type=pl.FunctionType.AIC, attrs={"split": pl.SplitMode.UP_DOWN})
            def main(self, x: pl.Tensor[[16, 128], pl.FP16]) -> pl.Tensor[[16, 128], pl.FP16]:
                return x

        ir.assert_structural_equal(A, B)

    def test_structural_inequality_different_split(self):
        @pl.program
        class A:
            @pl.function(type=pl.FunctionType.AIC, attrs={"split": pl.SplitMode.UP_DOWN})
            def main(self, x: pl.Tensor[[16, 128], pl.FP16]) -> pl.Tensor[[16, 128], pl.FP16]:
                return x

        @pl.program
        class B:
            @pl.function(type=pl.FunctionType.AIC, attrs={"split": pl.SplitMode.LEFT_RIGHT})
            def main(self, x: pl.Tensor[[16, 128], pl.FP16]) -> pl.Tensor[[16, 128], pl.FP16]:
                return x

        with pytest.raises(Exception):
            ir.assert_structural_equal(A, B)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
