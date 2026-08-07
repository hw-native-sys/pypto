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

``pad_value`` is an ordinary positional-or-keyword parameter, so every spelling
must also work positionally: inside a ``@pl.function`` body the parser has to
resolve the enum and the numeric sugars for a positional slot the same way it
resolves them for ``pad_value=``.
"""

import math

import pypto.language as pl
import pytest
from pypto import ir
from pypto.ir.op._pad_value import normalize_pad_value
from pypto.language.parser.diagnostics.exceptions import InvalidOperationError

# Module-level closure constant — exercises the bare-``Name`` positional path,
# which resolves from the closure rather than from the IR variable scope.
CLOSURE_PAD = pl.PadValue.max


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


class TestNormalizePadValueUnwrapsConstants:
    """A positional literal reaches the IR builders as ``ConstInt`` / ``ConstFloat``.

    The DSL parser materializes positional constants into IR so they carry its
    chosen dtype, so ``fillpad(t, 0)`` hands ``normalize_pad_value`` an IR node
    rather than a Python number. It must normalize identically to the bare
    literal, and must report the *value* — not the node type — when rejecting.
    """

    @pytest.mark.parametrize(
        "const,expected",
        [
            (ir.ConstInt(0, ir.DataType.INDEX, ir.Span.unknown()), ir.PadValue.zero),
            (ir.ConstFloat(0.0, ir.DataType.FP32, ir.Span.unknown()), ir.PadValue.zero),
            (ir.ConstFloat(math.inf, ir.DataType.FP32, ir.Span.unknown()), ir.PadValue.max),
            (ir.ConstFloat(-math.inf, ir.DataType.FP32, ir.Span.unknown()), ir.PadValue.min),
        ],
    )
    def test_accepts_wrapped_literals(self, const, expected):
        assert normalize_pad_value(const) is expected

    def test_rejects_wrapped_int_naming_the_value(self):
        const = ir.ConstInt(7, ir.DataType.INDEX, ir.Span.unknown())
        with pytest.raises(ValueError, match="got 7") as exc:
            normalize_pad_value(const)
        # The old message named the IR node type, which told the user nothing
        # about the value they actually wrote.
        assert "ConstInt" not in str(exc.value)

    def test_rejects_wrapped_float_naming_the_value(self):
        const = ir.ConstFloat(3.14, ir.DataType.FP32, ir.Span.unknown())
        with pytest.raises(ValueError, match="got 3.14") as exc:
            normalize_pad_value(const)
        assert "ConstFloat" not in str(exc.value)


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


class TestFillpadPositionalPadValue:
    """A positional ``pad_value`` IRs identically to the keyword spelling.

    ``pad_value`` is positional-or-keyword, so ``pl.fillpad(x, PadValue.zero)``
    must mean exactly what ``pl.fillpad(x, pad_value=PadValue.zero)`` means.
    Each case uses a distinct class name: the DSL looks a program's source up by
    class name, so reusing one name across cases would silently re-parse the
    first body and make every assertion vacuous.
    """

    def test_positional_enum(self):
        @pl.program
        class PositionalEnum:
            @pl.function(type=pl.FunctionType.InCore)
            def main(self, x: pl.Tensor[[8, 32], pl.FP32]) -> pl.Tensor[[8, 32], pl.FP32]:
                y: pl.Tensor[[8, 32], pl.FP32] = pl.fillpad(x, pl.PadValue.zero)
                return y

        @pl.program
        class KeywordEnum:
            @pl.function(type=pl.FunctionType.InCore)
            def main(self, x: pl.Tensor[[8, 32], pl.FP32]) -> pl.Tensor[[8, 32], pl.FP32]:
                y: pl.Tensor[[8, 32], pl.FP32] = pl.fillpad(x, pad_value=pl.PadValue.zero)
                return y

        ir.assert_structural_equal(PositionalEnum, KeywordEnum)

    def test_positional_int_literal(self):
        @pl.program
        class PositionalZeroInt:
            @pl.function(type=pl.FunctionType.InCore)
            def main(self, x: pl.Tensor[[8, 32], pl.FP32]) -> pl.Tensor[[8, 32], pl.FP32]:
                y: pl.Tensor[[8, 32], pl.FP32] = pl.fillpad(x, 0)
                return y

        @pl.program
        class KeywordZeroInt:
            @pl.function(type=pl.FunctionType.InCore)
            def main(self, x: pl.Tensor[[8, 32], pl.FP32]) -> pl.Tensor[[8, 32], pl.FP32]:
                y: pl.Tensor[[8, 32], pl.FP32] = pl.fillpad(x, pad_value=pl.PadValue.zero)
                return y

        ir.assert_structural_equal(PositionalZeroInt, KeywordZeroInt)

    def test_positional_float_literal(self):
        @pl.program
        class PositionalZeroFloat:
            @pl.function(type=pl.FunctionType.InCore)
            def main(self, x: pl.Tensor[[8, 32], pl.FP32]) -> pl.Tensor[[8, 32], pl.FP32]:
                y: pl.Tensor[[8, 32], pl.FP32] = pl.fillpad(x, 0.0)
                return y

        @pl.program
        class KeywordZeroFloat:
            @pl.function(type=pl.FunctionType.InCore)
            def main(self, x: pl.Tensor[[8, 32], pl.FP32]) -> pl.Tensor[[8, 32], pl.FP32]:
                y: pl.Tensor[[8, 32], pl.FP32] = pl.fillpad(x, pad_value=pl.PadValue.zero)
                return y

        ir.assert_structural_equal(PositionalZeroFloat, KeywordZeroFloat)

    def test_positional_inf(self):
        @pl.program
        class PositionalInf:
            @pl.function(type=pl.FunctionType.InCore)
            def main(self, x: pl.Tensor[[8, 32], pl.FP32]) -> pl.Tensor[[8, 32], pl.FP32]:
                y: pl.Tensor[[8, 32], pl.FP32] = pl.fillpad(x, math.inf)
                return y

        @pl.program
        class KeywordMax:
            @pl.function(type=pl.FunctionType.InCore)
            def main(self, x: pl.Tensor[[8, 32], pl.FP32]) -> pl.Tensor[[8, 32], pl.FP32]:
                y: pl.Tensor[[8, 32], pl.FP32] = pl.fillpad(x, pad_value=pl.PadValue.max)
                return y

        ir.assert_structural_equal(PositionalInf, KeywordMax)

    def test_positional_negative_inf(self):
        @pl.program
        class PositionalNegInf:
            @pl.function(type=pl.FunctionType.InCore)
            def main(self, x: pl.Tensor[[8, 32], pl.FP32]) -> pl.Tensor[[8, 32], pl.FP32]:
                y: pl.Tensor[[8, 32], pl.FP32] = pl.fillpad(x, -math.inf)
                return y

        @pl.program
        class KeywordMin:
            @pl.function(type=pl.FunctionType.InCore)
            def main(self, x: pl.Tensor[[8, 32], pl.FP32]) -> pl.Tensor[[8, 32], pl.FP32]:
                y: pl.Tensor[[8, 32], pl.FP32] = pl.fillpad(x, pad_value=pl.PadValue.min)
                return y

        ir.assert_structural_equal(PositionalNegInf, KeywordMin)

    def test_positional_closure_name(self):
        @pl.program
        class PositionalClosureName:
            @pl.function(type=pl.FunctionType.InCore)
            def main(self, x: pl.Tensor[[8, 32], pl.FP32]) -> pl.Tensor[[8, 32], pl.FP32]:
                y: pl.Tensor[[8, 32], pl.FP32] = pl.fillpad(x, CLOSURE_PAD)
                return y

        @pl.program
        class KeywordClosureName:
            @pl.function(type=pl.FunctionType.InCore)
            def main(self, x: pl.Tensor[[8, 32], pl.FP32]) -> pl.Tensor[[8, 32], pl.FP32]:
                y: pl.Tensor[[8, 32], pl.FP32] = pl.fillpad(x, pad_value=CLOSURE_PAD)
                return y

        ir.assert_structural_equal(PositionalClosureName, KeywordClosureName)

    def test_positional_tile_fillpad(self):
        @pl.program
        class PositionalTileFillpad:
            @pl.function(type=pl.FunctionType.InCore)
            def main(self, x: pl.Tensor[[8, 32], pl.FP32]) -> pl.Tensor[[8, 32], pl.FP32]:
                out = pl.create_tensor([8, 32], pl.FP32)
                t = pl.load(x, [0, 0], [8, 32], target_memory=pl.MemorySpace.Vec)
                p = pl.tile.fillpad(t, pl.PadValue.min)
                pl.store(p, [0, 0], out)
                return out

        @pl.program
        class KeywordTileFillpad:
            @pl.function(type=pl.FunctionType.InCore)
            def main(self, x: pl.Tensor[[8, 32], pl.FP32]) -> pl.Tensor[[8, 32], pl.FP32]:
                out = pl.create_tensor([8, 32], pl.FP32)
                t = pl.load(x, [0, 0], [8, 32], target_memory=pl.MemorySpace.Vec)
                p = pl.tile.fillpad(t, pad_value=pl.PadValue.min)
                pl.store(p, [0, 0], out)
                return out

        ir.assert_structural_equal(PositionalTileFillpad, KeywordTileFillpad)

    def test_positional_fillpad_expand(self):
        @pl.program
        class PositionalExpand:
            @pl.function(type=pl.FunctionType.InCore)
            def main(self, x: pl.Tensor[[8, 32], pl.FP32]) -> pl.Tensor[[16, 32], pl.FP32]:
                y: pl.Tensor[[16, 32], pl.FP32] = pl.fillpad_expand(x, [16, 32], pl.PadValue.max)
                return y

        @pl.program
        class KeywordExpand:
            @pl.function(type=pl.FunctionType.InCore)
            def main(self, x: pl.Tensor[[8, 32], pl.FP32]) -> pl.Tensor[[16, 32], pl.FP32]:
                y: pl.Tensor[[16, 32], pl.FP32] = pl.fillpad_expand(x, [16, 32], pad_value=pl.PadValue.max)
                return y

        ir.assert_structural_equal(PositionalExpand, KeywordExpand)


class TestFillpadPositionalRejection:
    """An invalid positional ``pad_value`` still raises, naming the value.

    The pre-fix message reported the parser's IR node type ("got ConstInt"),
    which named neither the value the user wrote nor anything actionable.
    """

    def test_positional_arbitrary_int_names_the_value(self):
        with pytest.raises(InvalidOperationError, match="got 7") as exc:

            @pl.program
            class BadPositionalInt:
                @pl.function(type=pl.FunctionType.InCore)
                def main(self, x: pl.Tensor[[8, 32], pl.FP32]) -> pl.Tensor[[8, 32], pl.FP32]:
                    y: pl.Tensor[[8, 32], pl.FP32] = pl.fillpad(x, 7)
                    return y

        assert "ConstInt" not in str(exc.value)

    def test_positional_arbitrary_float_names_the_value(self):
        with pytest.raises(InvalidOperationError, match="got 3.14") as exc:

            @pl.program
            class BadPositionalFloat:
                @pl.function(type=pl.FunctionType.InCore)
                def main(self, x: pl.Tensor[[8, 32], pl.FP32]) -> pl.Tensor[[8, 32], pl.FP32]:
                    y: pl.Tensor[[8, 32], pl.FP32] = pl.fillpad(x, 3.14)
                    return y

        assert "ConstFloat" not in str(exc.value)

    def test_positional_null_enum(self):
        with pytest.raises(InvalidOperationError, match="fillpad pad_value"):

            @pl.program
            class BadPositionalNull:
                @pl.function(type=pl.FunctionType.InCore)
                def main(self, x: pl.Tensor[[8, 32], pl.FP32]) -> pl.Tensor[[8, 32], pl.FP32]:
                    y: pl.Tensor[[8, 32], pl.FP32] = pl.fillpad(x, pl.PadValue.null)
                    return y


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
