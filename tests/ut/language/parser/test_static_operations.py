# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for pl.static_print() and pl.static_assert()."""

import pypto.language as pl
import pytest
from pypto import ir
from pypto.language.parser.diagnostics import (
    ParserError,
    ParserSyntaxError,
    UnsupportedFeatureError,
    concise_error_message,
)


class TestStaticPrint:
    """Tests for pl.static_print() parse-time printing."""

    def test_static_print_variable(self, capsys):
        """Test that static_print prints variable name and type."""

        @pl.function
        def func(x: pl.Tensor[[128, 64], pl.FP16]) -> pl.Tensor[[128, 64], pl.FP16]:
            pl.static_print(x)
            return x

        captured = capsys.readouterr()
        assert "x: pl.Tensor[[128, 64], pl.FP16]" in captured.out

    def test_static_print_string_label(self, capsys):
        """Test that string arguments are printed as-is."""

        @pl.function
        def func(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            pl.static_print("my label")
            return x

        captured = capsys.readouterr()
        assert "]: my label\n" in captured.out

    def test_static_print_multiple_args(self, capsys):
        """Test static_print with mixed string and variable args."""

        @pl.function
        def func(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            pl.static_print("input:", x)
            return x

        captured = capsys.readouterr()
        assert "]: input: x: pl.Tensor[[64], pl.FP32]\n" in captured.out

    def test_static_print_const(self, capsys):
        """Test static_print with a constant value."""

        @pl.function
        def func(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            pl.static_print(42)
            return x

        captured = capsys.readouterr()
        assert "]: 42: pl.Scalar[pl.INDEX]\n" in captured.out

    def test_static_print_fstring_with_variable(self, capsys):
        """Test that static_print supports f-strings with IR variables."""

        @pl.function
        def func(x: pl.Tensor[[128, 64], pl.FP16]) -> pl.Tensor[[128, 64], pl.FP16]:
            pl.static_print(f"input: {x}")
            return x

        captured = capsys.readouterr()
        assert "]: input: x: pl.Tensor[[128, 64], pl.FP16]\n" in captured.out

    def test_static_print_fstring_with_constant(self, capsys):
        """Test that static_print supports f-strings with constants."""

        @pl.function
        def func(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            pl.static_print(f"value: {42}")
            return x

        captured = capsys.readouterr()
        assert "]: value: 42: pl.Scalar[pl.INDEX]\n" in captured.out

    def test_static_print_fstring_multiple_placeholders(self, capsys):
        """Test f-string with multiple expression placeholders."""

        @pl.function
        def func(a: pl.Tensor[[64], pl.FP32], b: pl.Tensor[[32], pl.FP16]) -> pl.Tensor[[64], pl.FP32]:
            pl.static_print(f"a={a}, b={b}")
            return a

        captured = capsys.readouterr()
        output = captured.out
        assert "a=a: pl.Tensor[[64], pl.FP32]" in output
        assert "b=b: pl.Tensor[[32], pl.FP16]" in output

    def test_static_print_fstring_no_ir_generated(self):
        """Test that static_print with f-strings produces no IR."""

        @pl.function
        def with_print(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            pl.static_print(f"debug: {x}")
            result: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
            return result

        @pl.function
        def without_print(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            result: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
            return result

        ir.assert_structural_equal(with_print, without_print)

    def test_static_print_no_ir_generated(self):
        """Test that static_print produces no IR — programs are structurally equal."""

        @pl.function
        def with_print(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            pl.static_print(x)
            result: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
            return result

        @pl.function
        def without_print(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            result: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
            return result

        ir.assert_structural_equal(with_print, without_print)

    def test_static_print_no_args_error(self):
        """Test that static_print with no args raises error."""

        with pytest.raises(ParserSyntaxError, match="requires at least 1 argument"):

            @pl.function
            def func(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                pl.static_print()
                return x

    def test_static_print_via_text_parser(self, capsys):
        """Test that static_print works with pl.parse()."""
        code = """
import pypto.language as pl

@pl.function
def func(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    pl.static_print("check:", x)
    return x
"""
        func = pl.parse(code)
        assert isinstance(func, ir.Function)
        captured = capsys.readouterr()
        assert "]: check: x: pl.Tensor[[64], pl.FP32]\n" in captured.out

    def test_static_print_before_parse_error(self, capsys):
        """Test that static_print output appears even when parsing fails later."""

        with pytest.raises(ParserError):

            @pl.function
            def func(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                pl.static_print("debug:", x)
                pl.static_assert(False, "intentional failure")
                return x

        captured = capsys.readouterr()
        assert "]: debug: x: pl.Tensor[[64], pl.FP32]\n" in captured.out


class TestStaticAssert:
    """Tests for pl.static_assert() parse-time assertions."""

    def test_static_assert_true_passes(self):
        """Test that static_assert(True) succeeds."""

        @pl.function
        def func(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            pl.static_assert(True)
            return x

        assert isinstance(func, ir.Function)

    def test_static_assert_false_fails(self):
        """Test that static_assert(False) raises ParserError."""

        with pytest.raises(ParserError, match="static_assert failed"):

            @pl.function
            def func(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                pl.static_assert(False)
                return x

    def test_static_assert_with_message(self):
        """Test that custom message is included in error."""

        with pytest.raises(ParserError, match="shape mismatch"):

            @pl.function
            def func(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                pl.static_assert(False, "shape mismatch")
                return x

    def test_static_assert_nonzero_int_passes(self):
        """Test that static_assert(1) passes."""

        @pl.function
        def func(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            pl.static_assert(1)
            return x

        assert isinstance(func, ir.Function)

    def test_static_assert_zero_int_fails(self):
        """Test that static_assert(0) fails."""

        with pytest.raises(ParserError, match="static_assert failed"):

            @pl.function
            def func(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                pl.static_assert(0)
                return x

    def test_static_assert_closure_var(self):
        """Test static_assert with closure variable expression."""
        N = 64

        @pl.function
        def func(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            pl.static_assert(N > 32)
            return x

        assert isinstance(func, ir.Function)

    def test_static_assert_closure_var_fails(self):
        """Test static_assert fails when closure variable condition is false."""
        N = 10

        with pytest.raises(ParserError, match="static_assert failed"):

            @pl.function
            def func(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                pl.static_assert(N > 32)
                return x

    def test_static_assert_non_static_fails(self):
        """Test static_assert with non-compile-time IR variable raises error."""

        with pytest.raises(ParserError, match="compile-time evaluable"):

            @pl.function
            def func(x: pl.Scalar[pl.INT32]) -> pl.Scalar[pl.INT32]:
                pl.static_assert(x)
                return x

    def test_static_assert_no_ir_generated(self):
        """Test that static_assert produces no IR."""

        @pl.function
        def with_assert(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            pl.static_assert(True)
            result: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
            return result

        @pl.function
        def without_assert(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            result: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
            return result

        ir.assert_structural_equal(with_assert, without_assert)

    def test_static_assert_bad_arg_count(self):
        """Test static_assert with wrong number of args."""

        with pytest.raises(ParserSyntaxError, match="1 or 2 arguments"):

            @pl.function
            def func(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                pl.static_assert()
                return x

        with pytest.raises(ParserSyntaxError, match="1 or 2 arguments"):

            @pl.function
            def func2(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                pl.static_assert(True, "msg", "extra")
                return x

    def test_static_assert_via_text_parser(self):
        """Test that static_assert works with pl.parse()."""
        code = """
import pypto.language as pl

@pl.function
def func(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    pl.static_assert(True)
    return x
"""
        func = pl.parse(code)
        assert isinstance(func, ir.Function)

    def test_static_assert_via_text_parser_fails(self):
        """Test that static_assert failure works with pl.parse()."""
        code = """
import pypto.language as pl

@pl.function
def func(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    pl.static_assert(False, "bad condition")
    return x
"""
        with pytest.raises(ParserError, match="bad condition"):
            pl.parse(code)


class TestStaticPrintFstringErrors:
    """Tests for unsupported f-string features in static_print."""

    def test_fstring_conversion_rejected(self):
        """Test that f-string with !r conversion raises UnsupportedFeatureError."""

        with pytest.raises(UnsupportedFeatureError, match="conversion"):

            @pl.function
            def func(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                pl.static_print(f"{x!r}")
                return x

    def test_fstring_format_spec_rejected(self):
        """Test that f-string with format spec raises UnsupportedFeatureError."""

        with pytest.raises(UnsupportedFeatureError, match="format spec"):

            @pl.function
            def func(x: pl.Scalar[pl.INT32]) -> pl.Scalar[pl.INT32]:
                pl.static_print(f"{x:>10}")
                return x


class TestConciseErrorMessage:
    """Tests for the concise_error_message utility."""

    def test_plain_message_unchanged(self):
        """Test that a message without C++ noise is returned unchanged."""
        exc = ValueError("some error")
        assert concise_error_message(exc) == "some error"

    def test_strips_cpp_traceback(self):
        """Test stripping of C++ Traceback block."""
        msg = "user message\n\nC++ Traceback (most recent call last):\n File foo.cpp"
        exc = ValueError(msg)
        assert concise_error_message(exc) == "user message"

    def test_strips_no_stack_trace(self):
        """Test stripping of 'No stack trace available' block."""
        msg = "user message\n\nNo stack trace available.\n(Tip: Build with Debug)"
        exc = ValueError(msg)
        assert concise_error_message(exc) == "user message"

    def test_strips_check_failed(self):
        """Test stripping of CHECK macro suffix."""
        msg = "The op requires positive dim\nCheck failed: dim > 0 at src/foo.cpp:42"
        exc = ValueError(msg)
        assert concise_error_message(exc) == "The op requires positive dim"

    def test_strips_all_combined(self):
        """Test stripping when all noise types are present."""
        msg = (
            "user message\nCheck failed: x at f.cpp:1"
            "\n\nC++ Traceback (most recent call last):\n File bar.cpp"
        )
        exc = ValueError(msg)
        assert concise_error_message(exc) == "user message"

    def test_empty_message(self):
        """Test with empty exception message."""
        exc = ValueError("")
        assert concise_error_message(exc) == ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
