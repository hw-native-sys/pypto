# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for fillpad ``pad_value`` literal sugar.

The hardware only supports ``PadValue.zero`` / ``max`` / ``min``. The DSL
accepts the literal sugars ``0``, ``math.inf``, ``-math.inf`` as a friendlier
form; everything else must raise so a user is never silently given a different
fill value.
"""

import math

import pypto.language as pl
import pytest
from pypto import ir
from pypto.ir.op._pad_value import normalize_pad_value
from pypto.language.parser.diagnostics.exceptions import InvalidOperationError


class TestNormalizePadValueAccepts:
    """``normalize_pad_value`` accepts the enum and three numeric literals."""

    @pytest.mark.parametrize(
        "value,expected",
        [
            (ir.PadValue.zero, ir.PadValue.zero),
            (ir.PadValue.max, ir.PadValue.max),
            (ir.PadValue.min, ir.PadValue.min),
            (0, ir.PadValue.zero),
            (0.0, ir.PadValue.zero),
            (math.inf, ir.PadValue.max),
            (-math.inf, ir.PadValue.min),
        ],
    )
    def test_accepts(self, value, expected):
        assert normalize_pad_value(value) is expected


class TestNormalizePadValueRejects:
    """``normalize_pad_value`` rejects every other input with a clear hint."""

    @pytest.mark.parametrize(
        "value,exc_type",
        [
            (ir.PadValue.null, ValueError),
            (1, ValueError),
            (-1, ValueError),
            (42, ValueError),
            (3.14, ValueError),
            (-3.14, ValueError),
            (math.nan, ValueError),
            (True, TypeError),
            (False, TypeError),
            ("zero", TypeError),
            (None, TypeError),
            ([0], TypeError),
        ],
    )
    def test_rejects(self, value, exc_type):
        with pytest.raises(exc_type, match="fillpad pad_value"):
            normalize_pad_value(value)


class TestFillpadSugarMatchesEnum:
    """End-to-end: ``pl.fillpad`` with sugar IRs identically to the enum form."""

    @pytest.mark.parametrize(
        "literal,enum",
        [
            (0, ir.PadValue.zero),
            (math.inf, ir.PadValue.max),
            (-math.inf, ir.PadValue.min),
        ],
    )
    def test_tensor_fillpad(self, literal, enum):
        @pl.program
        class Sugared:
            @pl.function(type=pl.FunctionType.InCore)
            def main(self, x: pl.Tensor[[8, 32], pl.FP32]) -> pl.Tensor[[8, 32], pl.FP32]:
                y: pl.Tensor[[8, 32], pl.FP32] = pl.fillpad(x, pad_value=literal)
                return y

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main(self, x: pl.Tensor[[8, 32], pl.FP32]) -> pl.Tensor[[8, 32], pl.FP32]:
                y: pl.Tensor[[8, 32], pl.FP32] = pl.fillpad(x, pad_value=enum)
                return y

        ir.assert_structural_equal(Sugared, Expected)


class TestFillpadEndToEndRejection:
    """Invalid pad_value inside @pl.function bodies surfaces at parse time.

    The parser wraps the underlying ``TypeError`` / ``ValueError`` from
    ``normalize_pad_value`` in an ``InvalidOperationError`` (its standard
    behavior for any exception raised by an op builder), but the original
    hint text is preserved in the message so users still see the explanation.
    """

    def test_pl_fillpad_rejects_arbitrary_int(self):
        with pytest.raises(InvalidOperationError, match="fillpad pad_value"):

            @pl.program
            class Bad:
                @pl.function(type=pl.FunctionType.InCore)
                def main(self, x: pl.Tensor[[8, 32], pl.FP32]) -> pl.Tensor[[8, 32], pl.FP32]:
                    y: pl.Tensor[[8, 32], pl.FP32] = pl.fillpad(x, pad_value=7)
                    return y

    def test_pl_fillpad_rejects_arbitrary_float(self):
        with pytest.raises(InvalidOperationError, match="fillpad pad_value"):

            @pl.program
            class Bad:
                @pl.function(type=pl.FunctionType.InCore)
                def main(self, x: pl.Tensor[[8, 32], pl.FP32]) -> pl.Tensor[[8, 32], pl.FP32]:
                    y: pl.Tensor[[8, 32], pl.FP32] = pl.fillpad(x, pad_value=3.14)
                    return y

    def test_pl_fillpad_rejects_string(self):
        with pytest.raises(InvalidOperationError, match="fillpad pad_value"):

            @pl.program
            class Bad:
                @pl.function(type=pl.FunctionType.InCore)
                def main(self, x: pl.Tensor[[8, 32], pl.FP32]) -> pl.Tensor[[8, 32], pl.FP32]:
                    y: pl.Tensor[[8, 32], pl.FP32] = pl.fillpad(x, pad_value="zero")  # type: ignore[arg-type]
                    return y


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
