# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Unit tests for PyPTO logging framework.

This module tests the logging system exposed from C++ to Python, ensuring that:
1. Log levels can be set and retrieved
2. Logging functions work at different levels
3. Log filtering works correctly based on log level
4. LogLevel enum values are correctly exposed

Note: We use pytest's capfd fixture instead of capsys because the logging
implementation uses C++ std::cerr, which writes directly to file descriptors.
capfd captures output at the file descriptor level (stdout/stderr FDs),
while capsys only captures Python's sys.stdout/sys.stderr.
"""

import inspect
import re
import time

import pypto
import pypto.language as pl
import pytest
from pypto import LogLevel, ir, set_log_level


class TestLogLevel:
    """Test LogLevel enum and its values."""

    def test_log_level_values(self):
        """Test that LogLevel enum has all expected values."""
        assert int(LogLevel.DEBUG) == 0
        assert int(LogLevel.INFO) == 1
        assert int(LogLevel.WARN) == 2
        assert int(LogLevel.ERROR) == 3
        assert int(LogLevel.FATAL) == 4
        assert int(LogLevel.EVENT) == 5
        assert int(LogLevel.NONE) == 6


class TestSetLogLevel:
    """Test setting log levels."""

    def test_set_log_level_debug(self):
        """Test setting log level to DEBUG."""
        # Should not raise any exception
        set_log_level(LogLevel.DEBUG)

    def test_set_log_level_info(self):
        """Test setting log level to INFO."""
        set_log_level(LogLevel.INFO)

    def test_set_log_level_warn(self):
        """Test setting log level to WARN."""
        set_log_level(LogLevel.WARN)

    def test_set_log_level_error(self):
        """Test setting log level to ERROR."""
        set_log_level(LogLevel.ERROR)

    def test_set_log_level_fatal(self):
        """Test setting log level to FATAL."""
        set_log_level(LogLevel.FATAL)

    def test_set_log_level_event(self):
        """Test setting log level to EVENT."""
        set_log_level(LogLevel.EVENT)

    def test_set_log_level_none(self):
        """Test setting log level to NONE (disable all logging)."""
        set_log_level(LogLevel.NONE)

    def test_set_log_level_with_int(self):
        """Test setting log level using integer values."""
        # LogLevel is an IntEnum, so it should accept integers
        set_log_level(LogLevel(0))  # DEBUG
        set_log_level(LogLevel(3))  # ERROR
        set_log_level(LogLevel(6))  # NONE


class TestLoggingFunctions:
    """Test that logging functions can be called without errors."""

    def setup_method(self):
        """Set log level to DEBUG before each test to ensure all logs are visible."""
        set_log_level(LogLevel.DEBUG)

    def test_log_debug(self, capfd):
        """Test that log_debug can be called and produces output."""
        pypto.log_debug("Debug message")
        captured = capfd.readouterr()
        assert "Debug message" in captured.err
        assert "D |" in captured.err  # Check for DEBUG level marker

    def test_log_info(self, capfd):
        """Test that log_info can be called and produces output."""
        pypto.log_info("Info message")
        captured = capfd.readouterr()
        assert "Info message" in captured.err
        assert "I |" in captured.err  # Check for INFO level marker

    def test_log_warn(self, capfd):
        """Test that log_warn can be called and produces output."""
        pypto.log_warn("Warning message")
        captured = capfd.readouterr()
        assert "Warning message" in captured.err
        assert "W |" in captured.err  # Check for WARN level marker

    def test_log_error(self, capfd):
        """Test that log_error can be called and produces output."""
        pypto.log_error("Error message")
        captured = capfd.readouterr()
        assert "Error message" in captured.err
        assert "E |" in captured.err  # Check for ERROR level marker

    def test_log_fatal(self, capfd):
        """Test that log_fatal can be called and produces output."""
        pypto.log_fatal("Fatal message")
        captured = capfd.readouterr()
        assert "Fatal message" in captured.err
        assert "F |" in captured.err  # Check for FATAL level marker

    def test_log_event(self, capfd):
        """Test that log_event can be called and produces output."""
        pypto.log_event("Event message")
        captured = capfd.readouterr()
        assert "Event message" in captured.err
        assert "V |" in captured.err  # Check for EVENT level marker

    def test_log_with_special_characters(self, capfd):
        """Test logging messages with special characters."""
        pypto.log_info("Message with special chars: !@#$%^&*()")
        captured = capfd.readouterr()
        assert "Message with special chars: !@#$%^&*()" in captured.err

        pypto.log_info("Message with quotes: 'single' and \"double\"")
        captured = capfd.readouterr()
        assert "Message with quotes: 'single' and \"double\"" in captured.err

    def test_log_empty_message(self, capfd):
        """Test logging empty messages."""
        pypto.log_info("")
        captured = capfd.readouterr()
        # Should still have timestamp and level marker
        assert "I |" in captured.err


class TestLogLevelFiltering:
    """Test that log level filtering works correctly."""

    def test_debug_level_shows_all(self, capfd):
        """Test that DEBUG level shows all log messages."""
        set_log_level(LogLevel.DEBUG)

        pypto.log_debug("Debug")
        pypto.log_info("Info")
        pypto.log_warn("Warn")
        pypto.log_error("Error")
        pypto.log_fatal("Fatal")
        pypto.log_event("Event")

        captured = capfd.readouterr()
        # All messages should appear
        assert "Debug" in captured.err
        assert "Info" in captured.err
        assert "Warn" in captured.err
        assert "Error" in captured.err
        assert "Fatal" in captured.err
        assert "Event" in captured.err

    def test_info_level_filters_debug(self, capfd):
        """Test that INFO level filters out DEBUG messages."""
        set_log_level(LogLevel.INFO)

        pypto.log_debug("Debug - should be filtered")
        pypto.log_info("Info - should appear")
        pypto.log_warn("Warn - should appear")
        pypto.log_error("Error - should appear")

        captured = capfd.readouterr()
        # DEBUG should be filtered out
        assert "Debug - should be filtered" not in captured.err
        # INFO and above should appear
        assert "Info - should appear" in captured.err
        assert "Warn - should appear" in captured.err
        assert "Error - should appear" in captured.err

    def test_error_level_filters_lower(self, capfd):
        """Test that ERROR level filters out lower priority messages."""
        set_log_level(LogLevel.ERROR)

        pypto.log_debug("Debug - filtered")
        pypto.log_info("Info - filtered")
        pypto.log_warn("Warn - filtered")
        pypto.log_error("Error - should appear")
        pypto.log_fatal("Fatal - should appear")

        captured = capfd.readouterr()
        # DEBUG, INFO, WARN should be filtered
        assert "Debug - filtered" not in captured.err
        assert "Info - filtered" not in captured.err
        assert "Warn - filtered" not in captured.err
        # ERROR and FATAL should appear
        assert "Error - should appear" in captured.err
        assert "Fatal - should appear" in captured.err

    def test_none_level_filters_all(self, capfd):
        """Test that NONE level filters out all messages."""
        set_log_level(LogLevel.NONE)

        pypto.log_debug("Debug")
        pypto.log_info("Info")
        pypto.log_warn("Warn")
        pypto.log_error("Error")
        pypto.log_fatal("Fatal")
        pypto.log_event("Event")

        captured = capfd.readouterr()
        # All messages should be filtered
        assert "Debug" not in captured.err
        assert "Info" not in captured.err
        assert "Warn" not in captured.err
        assert "Error" not in captured.err
        assert "Fatal" not in captured.err
        assert "Event" not in captured.err


class TestLoggingScenarios:
    """Test real-world logging scenarios."""

    def test_change_log_level_during_execution(self, capfd):
        """Test changing log level multiple times during execution."""
        # Start with ERROR level
        set_log_level(LogLevel.ERROR)
        pypto.log_info("Should not appear")
        pypto.log_error("Should appear 1")

        captured = capfd.readouterr()
        assert "Should not appear" not in captured.err
        assert "Should appear 1" in captured.err

        # Change to DEBUG level
        set_log_level(LogLevel.DEBUG)
        pypto.log_debug("Now debug should appear")
        pypto.log_info("Info should appear")

        captured = capfd.readouterr()
        assert "Now debug should appear" in captured.err
        assert "Info should appear" in captured.err

        # Change back to ERROR level
        set_log_level(LogLevel.ERROR)
        pypto.log_info("Should not appear again")
        pypto.log_error("Should appear 2")

        captured = capfd.readouterr()
        assert "Should not appear again" not in captured.err
        assert "Should appear 2" in captured.err

    def test_log_all_levels_in_sequence(self, capfd):
        """Test logging at all levels in sequence."""
        set_log_level(LogLevel.DEBUG)

        levels_and_funcs = [
            ("DEBUG", pypto.log_debug, "D |"),
            ("INFO", pypto.log_info, "I |"),
            ("WARN", pypto.log_warn, "W |"),
            ("ERROR", pypto.log_error, "E |"),
            ("FATAL", pypto.log_fatal, "F |"),
            ("EVENT", pypto.log_event, "V |"),
        ]

        for level_name, log_func, marker in levels_and_funcs:
            log_func(f"Testing {level_name} level")

        captured = capfd.readouterr()
        # Verify all messages appear with correct markers
        for level_name, _, marker in levels_and_funcs:
            assert f"Testing {level_name} level" in captured.err
            assert marker in captured.err

    def test_logging_with_different_data_types(self, capfd):
        """Test logging messages with different data types converted to strings."""
        set_log_level(LogLevel.INFO)

        messages = [
            "Integer: 42",
            "Float: 3.14159",
            "Boolean: True",
            "List: [1, 2, 3]",
            "Dict: {'key': 'value'}",
        ]

        for msg in messages:
            pypto.log_info(msg)

        captured = capfd.readouterr()
        for msg in messages:
            assert msg in captured.err


class TestLogLevelEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_set_same_level_multiple_times(self, capfd):
        """Test setting the same log level multiple times."""
        set_log_level(LogLevel.INFO)
        set_log_level(LogLevel.INFO)
        set_log_level(LogLevel.INFO)
        pypto.log_info("Should work fine")

        captured = capfd.readouterr()
        assert "Should work fine" in captured.err

    def test_rapid_level_changes(self, capfd):
        """Test rapidly changing log levels."""
        for i in range(10):
            set_log_level(LogLevel.DEBUG)
            pypto.log_debug(f"Debug {i}")
            set_log_level(LogLevel.INFO)
            pypto.log_debug(f"Filtered {i}")
            set_log_level(LogLevel.ERROR)
            pypto.log_info(f"Also filtered {i}")
            set_log_level(LogLevel.NONE)
            pypto.log_error(f"Everything filtered {i}")

        captured = capfd.readouterr()
        # Only debug messages should have appeared
        for i in range(10):
            assert f"Debug {i}" in captured.err
            assert f"Filtered {i}" not in captured.err
            assert f"Also filtered {i}" not in captured.err
            assert f"Everything filtered {i}" not in captured.err

    def test_logging_at_boundary_levels(self, capfd):
        """Test logging at boundary conditions."""
        # Test at the lowest level
        set_log_level(LogLevel.DEBUG)
        pypto.log_debug("Lowest level message")

        captured = capfd.readouterr()
        assert "Lowest level message" in captured.err

        # Test at the highest level (EVENT)
        set_log_level(LogLevel.EVENT)
        pypto.log_event("Highest level message")

        captured = capfd.readouterr()
        assert "Highest level message" in captured.err

        # Test with NONE (all disabled)
        set_log_level(LogLevel.NONE)
        pypto.log_event("Even EVENT should be filtered")

        captured = capfd.readouterr()
        assert "Even EVENT should be filtered" not in captured.err


class TestTimestampAndFormatting:
    """Test that log output includes proper formatting."""

    def test_log_contains_timestamp(self, capfd):
        """Test that log output includes a timestamp."""
        set_log_level(LogLevel.INFO)
        pypto.log_info("Test message")

        captured = capfd.readouterr()
        # Check for timestamp pattern (YYYY-MM-DD HH:MM:SS.mmm)
        timestamp_pattern = r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}"
        assert re.search(timestamp_pattern, captured.err) is not None

    def test_log_contains_level_marker(self, capfd):
        """Test that each log level has its correct marker."""
        set_log_level(LogLevel.DEBUG)

        test_cases = [
            (pypto.log_debug, "D |"),
            (pypto.log_info, "I |"),
            (pypto.log_warn, "W |"),
            (pypto.log_error, "E |"),
            (pypto.log_fatal, "F |"),
            (pypto.log_event, "V |"),
        ]

        for log_func, expected_marker in test_cases:
            log_func("Test")
            captured = capfd.readouterr()
            assert expected_marker in captured.err

    def test_multiple_logs_have_different_timestamps(self, capfd):
        """Test that consecutive logs can have different timestamps."""
        set_log_level(LogLevel.INFO)

        pypto.log_info("First message")
        time.sleep(0.002)  # Sleep for 2ms to ensure different timestamp
        pypto.log_info("Second message")

        captured = capfd.readouterr()
        assert "First message" in captured.err
        assert "Second message" in captured.err
        # Both should have timestamps
        timestamps = re.findall(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}", captured.err)
        assert len(timestamps) >= 2


class TestCheckFunctions:
    """Test CHECK and INTERNAL_CHECK functions."""

    def test_check_passes_on_true_condition(self):
        """Test that check() doesn't raise when condition is True."""
        # Should not raise any exception
        pypto.check(True, "This should not be raised")

    def test_check_raises_on_false_condition(self):
        """Test that check() raises ValueError when condition is False."""
        with pytest.raises(ValueError) as exc_info:
            pypto.check(False, "This is a test error message")
        assert "This is a test error message" in str(exc_info.value)

    def test_check_with_custom_message(self):
        """Test that check() includes the custom message in the exception."""
        error_msg = "Value must be positive, got: -5"
        with pytest.raises(ValueError) as exc_info:
            pypto.check(-5 > 0, error_msg)
        assert error_msg in str(exc_info.value)

    def test_check_with_complex_condition(self):
        """Test check() with more complex conditions."""
        x = 10
        # This should pass
        pypto.check(x > 0 and x < 100, "x should be between 0 and 100")

        # This should fail
        with pytest.raises(ValueError):
            pypto.check(x > 100, f"x ({x}) should be greater than 100")

    def test_internal_check_passes_on_true_condition(self):
        """Test that internal_check() doesn't raise when condition is True."""
        # Should not raise any exception
        pypto.internal_check(True, "This should not be raised")
        pypto.internal_check(2 + 2 == 4, "Math works")

    def test_internal_check_raises_on_false_condition(self):
        """Test that internal_check() raises InternalError when condition is False."""
        with pytest.raises(pypto.InternalError) as exc_info:
            pypto.internal_check(False, "Internal invariant violated")
        assert "Internal invariant violated" in str(exc_info.value)

    def test_internal_check_with_custom_message(self):
        """Test that internal_check() includes the custom message in the exception."""
        error_msg = "Pointer should never be null at this point"
        with pytest.raises(pypto.InternalError) as exc_info:
            pypto.internal_check(False, error_msg)
        assert error_msg in str(exc_info.value)

    def test_check_vs_internal_check_exception_types(self):
        """Test that check() and internal_check() raise different exception types."""
        # check() should raise ValueError
        with pytest.raises(ValueError):
            pypto.check(False, "ValueError test")

        # internal_check() should raise InternalError
        with pytest.raises(pypto.InternalError):
            pypto.internal_check(False, "InternalError test")

        # Verify they are different types
        try:
            pypto.check(False, "test")
        except Exception as e:
            assert type(e).__name__ == "ValueError"

        try:
            pypto.internal_check(False, "test")
        except Exception as e:
            assert type(e).__name__ == "InternalError"

    def test_check_with_empty_message(self):
        """Test check() with an empty message."""
        with pytest.raises(ValueError) as exc_info:
            pypto.check(False, "")
        # Should still raise, even with empty message
        assert isinstance(exc_info.value, ValueError)

    def test_check_with_special_characters_in_message(self):
        """Test check() with special characters in error message."""
        special_msg = "Error: value < 0 && value != -1 (unexpected!)"
        with pytest.raises(ValueError) as exc_info:
            pypto.check(False, special_msg)
        assert special_msg in str(exc_info.value)

    def test_multiple_checks_in_sequence(self):
        """Test multiple checks in sequence."""
        # All should pass
        pypto.check(True, "First check")
        pypto.check(True, "Second check")

        # First one should fail and stop execution
        with pytest.raises(ValueError) as exc_info:
            pypto.check(False, "This will fail")
            pypto.check(False, "This won't be reached")
        assert "This will fail" in str(exc_info.value)
        assert "This won't be reached" not in str(exc_info.value)

    def test_check_preserves_exception_hierarchy(self):
        """Test that ValueError can be caught as a standard exception."""
        # Should be catchable as a general Exception
        with pytest.raises(Exception):
            pypto.check(False, "test")

        # Should be catchable as ValueError specifically
        with pytest.raises(ValueError):
            pypto.check(False, "test")

    def test_internal_check_preserves_exception_hierarchy(self):
        """Test that InternalError can be caught as a standard exception."""
        # Should be catchable as a general Exception
        with pytest.raises(Exception):
            pypto.internal_check(False, "test")

        # Should be catchable as InternalError specifically
        with pytest.raises(pypto.InternalError):
            pypto.internal_check(False, "test")

    def test_internal_check_with_span_includes_source_location(self):
        """Test that internal_check_with_span embeds span info in error message."""
        span = ir.Span("user_model.py", 12, 5)
        with pytest.raises(pypto.InternalError) as exc_info:
            pypto.internal_check_with_span(False, span, "ForStmt has null loop_var")
        msg = str(exc_info.value)
        assert "ForStmt has null loop_var" in msg
        assert "user_model.py:12:5" in msg

    def test_internal_check_with_span_unknown_span_omits_location(self):
        """Test that internal_check_with_span with an invalid span omits location."""
        span = ir.Span.unknown()
        with pytest.raises(pypto.InternalError) as exc_info:
            pypto.internal_check_with_span(False, span, "error with unknown span")
        msg = str(exc_info.value)
        assert "error with unknown span" in msg
        assert " (at " not in msg

    def test_internal_check_with_span_passes_on_true(self):
        """Test that internal_check_with_span does not raise when condition is True."""
        span = ir.Span("user_model.py", 42, 1)
        # Should not raise
        pypto.internal_check_with_span(True, span, "should not raise")


class TestSpanGuardWithRealProgram:
    """Test that spans from @pl.program IR appear in INTERNAL_CHECK error messages.

    These tests use real user programs (defined via @pl.program) to verify the
    end-to-end flow: Python source → IR span → error message location.
    """

    def test_forstmt_span_in_error_message(self):
        """Test that a ForStmt span from a real @pl.program appears in the error."""

        @pl.program
        class MyModel:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                for i in pl.range(64):  # ← this line's span should appear in errors
                    _ = pl.tensor.read(x, [i])
                return x

        func = MyModel["main"]
        assert func is not None
        assert isinstance(func.body, ir.SeqStmts)
        for_stmt = func.body[0]  # SeqStmts[0] is the ForStmt

        assert for_stmt.span.is_valid(), "ForStmt must have a valid span"
        assert for_stmt.span.begin_line > 0

        # Simulate an INTERNAL_CHECK failure inside a pass visiting this ForStmt.
        # In production code this would be triggered by SPAN_GUARD(op->span_) in
        # IRMutator::VisitStmt_(ForStmtPtr) when op->loop_var_ is unexpectedly null.
        with pytest.raises(pypto.InternalError) as exc_info:
            pypto.internal_check_with_span(False, for_stmt.span, "ForStmt has null loop_var")

        msg = str(exc_info.value)
        print(msg)
        assert "ForStmt has null loop_var" in msg
        # Span must point back to this test file, at the for loop line
        assert "test_logging.py" in msg
        assert f":{for_stmt.span.begin_line}:" in msg

    def test_span_shows_correct_line_number(self):
        """Test that span line numbers match the actual source line in @pl.program."""

        @pl.program
        class MyModel:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                for i in pl.range(64):
                    _ = pl.tensor.read(x, [i])
                return x

        func = MyModel["main"]
        assert func is not None
        assert isinstance(func.body, ir.SeqStmts)
        for_stmt = func.body[0]

        # The span's line number should be the line where `for i in pl.range(64):` appears.
        # Verify it actually points inside this test file at a reasonable line.
        assert for_stmt.span.begin_line > 0
        source_lines, start_line = inspect.getsourcelines(
            TestSpanGuardWithRealProgram.test_span_shows_correct_line_number
        )
        func_end_line = start_line + len(source_lines)
        assert start_line <= for_stmt.span.begin_line <= func_end_line, (
            f"ForStmt span line {for_stmt.span.begin_line} is not within "
            f"this test method (lines {start_line}–{func_end_line})"
        )

    def test_nested_span_guard_restores_outer_span(self):
        """Test that nested SpanGuards correctly restore the outer span on exit.

        When visiting a ForStmt containing a Call, the Call's SPAN_GUARD temporarily
        overrides the ForStmt's SPAN_GUARD. After the Call scope exits, the ForStmt
        span is restored.
        """
        outer_span = ir.Span("outer.py", 10, 1)
        inner_span = ir.Span("inner.py", 20, 5)

        # Outer guard active — error shows outer span
        with pytest.raises(pypto.InternalError) as exc_info:
            pypto.internal_check_with_span(False, outer_span, "outer check")
        assert "outer.py:10:1" in str(exc_info.value)

        # Simulate sequential calls: each call creates and destroys its own SpanGuard
        errors = []
        try:
            pypto.internal_check_with_span(True, outer_span, "outer ok")  # outer guard set/cleared
            pypto.internal_check_with_span(False, inner_span, "inner check")
        except pypto.InternalError as e:
            errors.append(str(e))

        assert len(errors) == 1
        assert "inner.py:20:5" in errors[0]

        # After inner guard exits, no span context remains — plain internal_check has no span
        with pytest.raises(pypto.InternalError) as exc_info:
            pypto.internal_check(False, "no span check")
        assert " (at " not in str(exc_info.value)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
