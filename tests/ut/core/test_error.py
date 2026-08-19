# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Unit tests for PyPTO error handling and reporting.

This module tests the error classes exposed from C++ to Python, ensuring that:
1. Errors are properly raised and caught
2. Stack traces are captured and included in error messages
3. Error types map correctly to Python's built-in exceptions
4. Error inheritance works as expected
"""

import re
import sys
from collections.abc import Callable

import pypto
import pytest
from pypto import testing

RaiseFn = Callable[[str], None]

_TRACE_HEADER = "C++ Traceback"
_NO_TRACE = "No stack trace available"

# Upstream libbacktrace only reads MH_EXECUTE / MH_DYLIB / MH_DSYM Mach-O files, and a CPython
# extension module is an MH_BUNDLE, so macOS never symbolizes and always takes the documented
# fallback. See docs/en/dev/02-error-handling.md.
_MACOS = sys.platform == "darwin"

# Matches the frame lines emitted by Backtrace::FormatStackTrace.
_FRAME_RE = re.compile(r'^ File "([^"]+)", line (\d+)$', re.MULTILINE)

# PyPTO's own C++ implementation — everything except the python/bindings/ entry points. CPython's
# sources (Python/, Objects/, Modules/, Include/) do not match: its include dir is capitalised.
_PYPTO_IMPL_RE = re.compile(r"(?:^|/)(?:src|include/pypto)/")


def _assert_has_trace(message: str) -> list[str]:
    """Assert *message* carries a stack trace and return the source files it names.

    The assertion is strict per platform rather than "trace *or* fallback": that disjunction is a
    tautology — GetFullMessage() always emits exactly one of the two — and would stay green even if
    symbolization broke completely. A Linux build without debug info fails here, which is the point.
    """
    if _MACOS:
        assert _NO_TRACE in message, f"expected the macOS fallback, got: {message}"
        return []

    assert _TRACE_HEADER in message, f"expected a C++ traceback, got: {message}"
    frames = _FRAME_RE.findall(message.split(_TRACE_HEADER, 1)[1])
    assert frames, f"traceback header present but no frames were symbolized: {message}"
    return [path for path, _ in frames]


def _assert_no_trace(message: str) -> None:
    """Assert *message* carries neither a stack trace nor the no-trace fallback."""
    assert _TRACE_HEADER not in message, f"unexpected traceback in user-facing error: {message}"
    assert _NO_TRACE not in message, f"unexpected no-trace notice in user-facing error: {message}"


class TestErrorTypes:
    """Test that different error types are raised correctly."""

    def test_value_error_type(self):
        """Test that ValueError is raised with correct type."""
        with pytest.raises(ValueError) as exc_info:
            testing.raise_value_error("test value error")

        assert "test value error" in str(exc_info.value)

    def test_type_error_type(self):
        """Test that TypeError is raised with correct type."""
        with pytest.raises(TypeError) as exc_info:
            testing.raise_type_error("test type error")

        assert "test type error" in str(exc_info.value)

    def test_runtime_error_type(self):
        """Test that RuntimeError is raised with correct type."""
        with pytest.raises(RuntimeError) as exc_info:
            testing.raise_runtime_error("test runtime error")

        assert "test runtime error" in str(exc_info.value)

    def test_not_implemented_error_type(self):
        """Test that NotImplementedError is raised with correct type."""
        with pytest.raises(NotImplementedError) as exc_info:
            testing.raise_not_implemented_error("test not implemented")

        assert "test not implemented" in str(exc_info.value)

    def test_index_error_type(self):
        """Test that IndexError is raised with correct type."""
        with pytest.raises(IndexError) as exc_info:
            testing.raise_index_error("test index error")

        assert "test index error" in str(exc_info.value)

    def test_generic_error_type(self):
        """Test that generic Error is raised with correct type."""
        with pytest.raises(pypto.Error) as exc_info:
            testing.raise_generic_error("test generic error")

        assert "test generic error" in str(exc_info.value)

    def test_assertion_error_type(self):
        """Test that AssertionError is raised with correct type."""
        with pytest.raises(AssertionError):
            testing.raise_assertion_error("test assertion error")

    def test_internal_error_type(self):
        """Test that InternalError is raised with correct type."""
        with pytest.raises(RuntimeError) as exc_info:
            testing.raise_internal_error("test internal error")

        assert "test internal error" in str(exc_info.value)


class TestErrorMessages:
    """Test that error messages are properly formatted and include necessary information."""

    def test_error_message_content(self):
        """Test that error messages contain the expected text."""
        with pytest.raises(ValueError) as exc_info:
            testing.raise_value_error("Custom error message")

        assert "Custom error message" in str(exc_info.value)

    def test_error_message_with_special_characters(self):
        """Test that error messages with special characters are handled correctly."""
        special_message = "Error with special chars: !@#$%^&*()"
        with pytest.raises(ValueError) as exc_info:
            testing.raise_value_error(special_message)

        assert special_message in str(exc_info.value)

    def test_error_message_with_numbers(self):
        """Test that error messages with numbers are handled correctly."""
        message = "Error code: 12345, value: 67890"
        with pytest.raises(RuntimeError) as exc_info:
            testing.raise_runtime_error(message)

        assert "12345" in str(exc_info.value)
        assert "67890" in str(exc_info.value)

    def test_multiline_error_message(self):
        """Test that multiline error messages are handled correctly."""
        message = "Line 1\nLine 2\nLine 3"
        with pytest.raises(TypeError) as exc_info:
            testing.raise_type_error(message)

        assert "Line 1" in str(exc_info.value)


@pytest.fixture
def no_backtrace_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin PTO_BACKTRACE to unset so default behaviour is tested regardless of the caller's env."""
    monkeypatch.delenv("PTO_BACKTRACE", raising=False)


class TestStackTraceVisibility:
    """Test which errors carry a C++ stack trace.

    Bug-class exceptions (the INTERNAL_CHECK family) always carry one; user errors (the CHECK
    family) only under PTO_BACKTRACE=1.
    """

    @pytest.mark.parametrize(
        "raise_fn",
        [
            testing.raise_value_error,
            testing.raise_type_error,
            testing.raise_runtime_error,
            testing.raise_index_error,
        ],
    )
    def test_user_error_omits_traceback_by_default(self, raise_fn: RaiseFn, no_backtrace_env: None):
        """A user error is just the message — internal frames are noise to the caller."""
        with pytest.raises((ValueError, TypeError, RuntimeError, IndexError)) as exc_info:
            raise_fn("user mistake")

        message = str(exc_info.value)
        assert message == "user mistake"
        _assert_no_trace(message)

    @pytest.mark.parametrize("raise_fn", [testing.raise_internal_error, testing.raise_assertion_error])
    def test_bug_error_always_includes_traceback(self, raise_fn: RaiseFn, no_backtrace_env: None):
        """The INTERNAL_CHECK family signals a PyPTO bug — the trace is the primary artefact."""
        with pytest.raises((pypto.InternalError, AssertionError)) as exc_info:
            raise_fn("invariant broken")

        message = str(exc_info.value)
        assert "invariant broken" in message
        _assert_has_trace(message)

    def test_user_error_includes_traceback_when_opted_in(self, monkeypatch: pytest.MonkeyPatch):
        """PTO_BACKTRACE=1 turns the trace back on for user errors."""
        monkeypatch.setenv("PTO_BACKTRACE", "1")

        with pytest.raises(ValueError) as exc_info:
            testing.raise_value_error("cannot reshape tile")

        message = str(exc_info.value)
        assert "cannot reshape tile" in message
        _assert_has_trace(message)

    def test_user_error_omits_traceback_when_opted_out(self, monkeypatch: pytest.MonkeyPatch):
        """Only the exact value "1" enables the trace."""
        monkeypatch.setenv("PTO_BACKTRACE", "0")

        with pytest.raises(ValueError) as exc_info:
            testing.raise_value_error("cannot reshape tile")

        _assert_no_trace(str(exc_info.value))


@pytest.mark.skipif(_MACOS, reason="macOS cannot symbolize a CPython MH_BUNDLE extension module")
class TestStackTraceContents:
    """Test that captured traces describe the real call path.

    These pin symbolization itself: they fail if the libbacktrace integration regresses, rather
    than silently degrading to the no-trace fallback.
    """

    def test_traceback_names_the_real_throw_site(self, no_backtrace_env: None):
        """The innermost frame must be the binding that threw."""
        with pytest.raises(pypto.InternalError) as exc_info:
            testing.raise_internal_error("invariant broken")

        files = _assert_has_trace(str(exc_info.value))
        assert any(f.endswith("python/bindings/modules/testing.cpp") for f in files), (
            f"expected the throwing binding in the trace, got: {files}"
        )

    def test_traceback_has_no_frames_from_pypto_internals(self, no_backtrace_env: None):
        """No PyPTO implementation file may appear in a trace that never entered one.

        raise_internal_error is reached straight from the interpreter through nanobind, so the
        binding is the only PyPTO source on the path — nothing under src/ or include/pypto/.
        Passing this module's own path to backtrace_create_state used to register its DWARF at the
        *interpreter's* load base, so CPython PCs resolved to whichever PyPTO line sat at the same
        offset; every fabricated frame therefore named a PyPTO implementation file.

        Interpreter frames (CPython's own Python/ceval.c, Objects/call.c, ...) are *not* filtered
        out: they are the genuine callers, and they appear whenever the running Python itself was
        built with debug info. Their presence is correct, so the assertion targets PyPTO sources
        rather than allowlisting the binding.
        """
        with pytest.raises(pypto.InternalError) as exc_info:
            testing.raise_internal_error("invariant broken")

        files = _assert_has_trace(str(exc_info.value))
        fabricated = [f for f in files if _PYPTO_IMPL_RE.search(f)]
        assert not fabricated, f"frames attributed to PyPTO internals not on the call path: {fabricated}"

    def test_traceback_omits_check_macro_infrastructure(self, no_backtrace_env: None):
        """The CHECK/INTERNAL_CHECK throw site in logging.h is noise, not a call-path frame."""
        with pytest.raises(pypto.InternalError) as exc_info:
            testing.raise_internal_error_with_span("boom", "kernel.py", 3, 5)

        files = _assert_has_trace(str(exc_info.value))
        assert not [f for f in files if f.endswith("core/logging.h")], (
            f"macro infrastructure leaked into the trace: {files}"
        )

    def test_different_errors_have_different_traces(self, monkeypatch: pytest.MonkeyPatch):
        """Distinct throw sites must produce distinct frames, not one cached trace."""
        monkeypatch.setenv("PTO_BACKTRACE", "1")

        with pytest.raises(ValueError) as value_info:
            testing.raise_value_error("error 1")
        with pytest.raises(TypeError) as type_info:
            testing.raise_type_error("error 2")

        value_message = str(value_info.value)
        type_message = str(type_info.value)
        assert "error 1" in value_message
        assert "error 2" in type_message
        _assert_has_trace(value_message)
        _assert_has_trace(type_message)

        value_trace = value_message.split(_TRACE_HEADER, 1)[1]
        type_trace = type_message.split(_TRACE_HEADER, 1)[1]
        assert value_trace != type_trace, "expected distinct throw sites to symbolize differently"


class TestErrorInheritance:
    """Test that error inheritance works correctly."""

    def test_value_error_is_exception(self):
        """Test that ValueError can be caught as Exception."""
        with pytest.raises(Exception):
            testing.raise_value_error("test")

    def test_type_error_is_exception(self):
        """Test that TypeError can be caught as Exception."""
        with pytest.raises(Exception):
            testing.raise_type_error("test")

    def test_runtime_error_is_exception(self):
        """Test that RuntimeError can be caught as Exception."""
        with pytest.raises(Exception):
            testing.raise_runtime_error("test")

    def test_index_error_is_exception(self):
        """Test that IndexError can be caught as Exception."""
        with pytest.raises(Exception):
            testing.raise_index_error("test")

    def test_assertion_error_is_exception(self):
        """Test that AssertionError can be caught as Exception."""
        with pytest.raises(Exception):
            testing.raise_assertion_error("test")

    def test_internal_error_is_exception(self):
        """Test that InternalError can be caught as Exception."""
        with pytest.raises(Exception):
            testing.raise_internal_error("test")


class TestErrorCatching:
    """Test various error catching scenarios."""

    def test_catch_specific_error(self):
        """Test that specific error types can be caught."""
        caught = False
        try:
            testing.raise_value_error("test")
        except ValueError:
            caught = True

        assert caught

    def test_catch_with_wrong_type_fails(self):
        """Test that catching with wrong type doesn't work."""
        with pytest.raises(ValueError):
            try:
                testing.raise_value_error("test")
            except TypeError:
                pass  # This should not catch the ValueError

    def test_multiple_error_types(self):
        """Test handling multiple different error types."""
        error_types = [
            (testing.raise_value_error, ValueError),
            (testing.raise_type_error, TypeError),
            (testing.raise_runtime_error, RuntimeError),
            (testing.raise_index_error, IndexError),
            (testing.raise_not_implemented_error, NotImplementedError),
            (testing.raise_assertion_error, AssertionError),
            (testing.raise_internal_error, RuntimeError),
        ]

        for raise_func, expected_type in error_types:
            with pytest.raises(expected_type):
                raise_func("test message")


class TestErrorContexts:
    """Test errors in various contexts."""

    def test_error_in_nested_calls(self):
        """Test that errors can be raised from nested function calls."""

        def level_3():
            testing.raise_runtime_error("nested error")

        def level_2():
            level_3()

        def level_1():
            level_2()

        with pytest.raises(RuntimeError) as exc_info:
            level_1()

        assert "nested error" in str(exc_info.value)

    def test_error_message_formatting(self):
        """Test that error messages are properly formatted."""
        test_cases = [
            "Simple message",
            "Message with 'quotes'",
            'Message with "double quotes"',
            "Message with\ttabs",
        ]

        for message in test_cases:
            with pytest.raises(ValueError) as exc_info:
                testing.raise_value_error(message)

            # The message should be preserved in some form
            assert len(str(exc_info.value)) > 0


class TestErrorEdgeCases:
    """Test edge cases and boundary conditions for error handling."""

    def test_empty_error_message(self):
        """Test that empty error messages are handled."""
        with pytest.raises(ValueError):
            testing.raise_value_error("")

    def test_very_long_error_message(self):
        """Test that very long error messages are handled."""
        long_message = "X" * 10000
        with pytest.raises(ValueError) as exc_info:
            testing.raise_value_error(long_message)

        assert "X" in str(exc_info.value)

    def test_error_with_null_characters(self):
        """Test error messages with null characters."""
        # Python strings don't allow null bytes in the middle,
        # but we can test with other control characters
        message = "Error\x00Test"  # This will be truncated at null
        try:
            with pytest.raises(ValueError):
                testing.raise_value_error(message)
        except Exception:
            # Some systems might handle this differently
            pass


class TestSpanInErrors:
    """Test that IR source span information is included in internal errors."""

    def test_internal_error_with_span_contains_source_location(self):
        """Test that InternalError with span includes the source location."""
        with pytest.raises(RuntimeError) as exc_info:
            testing.raise_internal_error_with_span("shape mismatch", "user_model.py", 42, 5)

        error_str = str(exc_info.value)
        assert "[user_model.py:42:5]" in error_str

    def test_internal_error_with_span_preserves_message(self):
        """Test that the user-provided error message is still present."""
        with pytest.raises(RuntimeError) as exc_info:
            testing.raise_internal_error_with_span("tensor rank error", "my_script.py", 10, 1)

        error_str = str(exc_info.value)
        assert "tensor rank error" in error_str
        assert "Check failed:" in error_str

    def test_internal_error_with_span_contains_cpp_location(self):
        """Test that C++ file/line info is also present alongside the span."""
        with pytest.raises(RuntimeError) as exc_info:
            testing.raise_internal_error_with_span("bad state", "example.py", 99, 3)

        error_str = str(exc_info.value)
        assert "[example.py:99:3]" in error_str
        assert "Check failed:" in error_str


class TestRethrowWithMessage:
    """Test `Error::RethrowWithMessage` — typed, trace-preserving message replacement.

    An intermediate frame often wants to add context to an in-flight error (the op registry
    appends the IR span to every type-deduction failure). Catching `const Error&` and
    constructing a fresh exception does that at the cost of collapsing the concrete type and
    replacing the original stack trace with the catcher's own. `RethrowWithMessage` exists so
    that context can be added without paying either.
    """

    _EXPECTED_PY_TYPE = {
        "ValueError": ValueError,
        "TypeError": TypeError,
        "RuntimeError": RuntimeError,
        "NotImplementedError": NotImplementedError,
        "IndexError": IndexError,
        "AssertionError": AssertionError,
        "InternalError": pypto.InternalError,
        "Error": pypto.Error,
    }

    @pytest.mark.parametrize("kind", list(_EXPECTED_PY_TYPE))
    def test_concrete_type_is_preserved(self, kind: str):
        """Each subclass rethrows as itself, not as its base or as ValueError."""
        with pytest.raises(self._EXPECTED_PY_TYPE[kind]):
            testing.rethrow_with_message(kind, "original", "replacement")

    @pytest.mark.parametrize("kind", list(_EXPECTED_PY_TYPE))
    def test_message_is_replaced(self, kind: str):
        with pytest.raises(self._EXPECTED_PY_TYPE[kind]) as exc_info:
            testing.rethrow_with_message(kind, "original", "replacement")

        # Bug-class errors append the traceback, which echoes the source line of each
        # frame - including `RaiseByKind(kind, original);`. Compare only the message.
        message = str(exc_info.value).split(_TRACE_HEADER, 1)[0]
        assert "replacement" in message
        assert "original" not in message

    @pytest.mark.skipif(_MACOS, reason="macOS cannot symbolize a CPython MH_BUNDLE extension module")
    @pytest.mark.parametrize("kind", ["InternalError", "AssertionError"])
    def test_bug_class_keeps_the_original_throw_site(self, kind: str, no_backtrace_env: None):
        """The trace keeps the frame that threw, below the frame that caught and rethrew.

        Both frames live in the binding module, so their *count* is what discriminates:
        RaiseByKind and rethrow_with_message contribute one each when the trace is carried
        over, while constructing a fresh exception in the catch block would leave only the
        latter - every frame below the catch having already been unwound.
        """
        with pytest.raises((pypto.InternalError, AssertionError)) as exc_info:
            testing.rethrow_with_message(kind, "original", "replacement")

        files = _assert_has_trace(str(exc_info.value))
        binding_frames = [f for f in files if f.endswith("python/bindings/modules/testing.cpp")]
        assert len(binding_frames) >= 2, (
            f"expected both the throw and rethrow frames, got {len(binding_frames)} in: {files}"
        )

    @pytest.mark.skipif(_MACOS, reason="macOS cannot symbolize a CPython MH_BUNDLE extension module")
    def test_direct_raise_has_one_frame_at_the_throw_site(self, no_backtrace_env: None):
        """Baseline for the test above: a plain raise contributes exactly one binding frame."""
        with pytest.raises(pypto.InternalError) as exc_info:
            testing.raise_internal_error("invariant broken")

        files = _assert_has_trace(str(exc_info.value))
        binding_frames = [f for f in files if f.endswith("python/bindings/modules/testing.cpp")]
        assert len(binding_frames) == 1, f"expected a single binding frame, got: {files}"

    def test_internal_error_is_not_a_value_error(self):
        """The regression this mechanism exists to prevent."""
        with pytest.raises(pypto.InternalError) as exc_info:
            testing.rethrow_with_message("InternalError", "original", "replacement")

        assert not isinstance(exc_info.value, ValueError)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
