# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for wrapping non-ParserError exceptions raised during parsing.

Validates that raw Python exceptions (ValueError, TypeError, etc.) raised by
op functions or IR builder calls are caught and re-raised as ParserError
subclasses with source location information, rather than escaping as raw
tracebacks.
"""

import pypto
import pypto.language as pl
import pytest
from pypto import ir
from pypto.language.op import tensor_ops as _dsl_tensor
from pypto.language.parser.diagnostics import (
    InvalidOperationError,
    ParserError,
    ParserTypeError,
)
from pypto.language.parser.diagnostics.renderer import ErrorRenderer


class TestOpErrorWrapping:
    """Tests that op function errors are wrapped as InvalidOperationError with span."""

    def test_tensor_cast_invalid_mode_in_function(self):
        """ValueError from tensor.cast gets wrapped with span in @pl.function."""
        with pytest.raises(InvalidOperationError, match="Invalid rounding mode") as exc_info:

            @pl.function
            def bad_cast(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.BF16]:
                result: pl.Tensor[[64], pl.BF16] = pl.tensor.cast(
                    x,
                    target_type=pl.BF16,
                    mode=99,
                )
                return result

        assert exc_info.value.span is not None

    def test_tensor_cast_invalid_mode_in_program(self):
        """ValueError from tensor.cast gets wrapped with span in @pl.program."""
        with pytest.raises(InvalidOperationError, match="Invalid rounding mode") as exc_info:

            @pl.program
            class BadCastProgram:
                @pl.function
                def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.BF16]:
                    result: pl.Tensor[[64], pl.BF16] = pl.tensor.cast(
                        x,
                        target_type=pl.BF16,
                        mode=99,
                    )
                    return result

        assert exc_info.value.span is not None

    def test_op_error_includes_operation_name(self):
        """Wrapped error message includes the operation name for context."""
        with pytest.raises(InvalidOperationError, match="tensor operation 'cast'"):

            @pl.function
            def bad_cast(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.BF16]:
                result: pl.Tensor[[64], pl.BF16] = pl.tensor.cast(
                    x,
                    target_type=pl.BF16,
                    mode="invalid_mode",  # type: ignore[arg-type]
                )
                return result

    def test_op_error_preserves_original_cause(self):
        """Wrapped error chains to the original exception via __cause__."""
        with pytest.raises(InvalidOperationError) as exc_info:

            @pl.function
            def bad_cast(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.BF16]:
                result: pl.Tensor[[64], pl.BF16] = pl.tensor.cast(
                    x,
                    target_type=pl.BF16,
                    mode="bad",  # type: ignore[arg-type]
                )
                return result

        assert exc_info.value.__cause__ is not None
        assert isinstance(exc_info.value.__cause__, ValueError)

    def test_parser_errors_not_double_wrapped(self):
        """ParserErrors from op dispatch are not re-wrapped."""
        with pytest.raises(InvalidOperationError, match=r"Unknown operation 'pl\.nonexistent_op'"):

            @pl.function
            def unknown(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                result: pl.Tensor[[64], pl.FP32] = pl.nonexistent_op(x)  # type: ignore
                return result


class TestBackendCheckMessageSanitized:
    """A C++ CHECK must not leak FatalLogger's tail into the user-facing diagnostic.

    Most ops type-check in C++. ``CHECK`` throws ``pypto::ValueError``, which surfaces in
    Python as a plain ``ValueError`` whose message still carries the tail
    ``FatalLogger::~FatalLogger`` appends: "Check failed: <C++ expr> at <abs path>.cpp:<line>".
    Left in, the renderer splices it into the bold ``Error:`` header, ahead of the ``-->``
    arrow that points at the user's own source.
    """

    @staticmethod
    def _mismatched_matmul():
        """Build a kernel whose only fault is FP16 x FP32 matmul operands.

        Drives the CHECK in src/ir/op/tile_ops/matmul.cpp -- the DSL wrapper does no dtype
        checking of its own, so validation happens in the backend type-deduction function.
        """

        @pl.function
        def bad(
            t1: pl.Tensor[[64, 64], pl.FP16],
            t2: pl.Tensor[[64, 64], pl.FP32],
            out: pl.Tensor[[64, 64], pl.FP32],
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            a: pl.Tile[[64, 64], pl.FP16] = pl.tile.load(t1, offsets=[0, 0], shapes=[64, 64])
            b: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(t2, offsets=[0, 0], shapes=[64, 64])
            c: pl.Tile[[64, 64], pl.FP32] = pl.tile.matmul(a, b)
            result: pl.Tensor[[64, 64], pl.FP32] = pl.tile.store(c, offsets=[0, 0], output_tensor=out)
            return result

        return bad

    def test_cpp_check_error_does_not_leak_check_failed_tail(self):
        """The header carries the op's own message -- not the C++ expression or path."""
        with pytest.raises(InvalidOperationError) as exc_info:
            self._mismatched_matmul()

        message = exc_info.value.message
        assert "Check failed" not in message
        assert ".cpp" not in message
        assert "src/ir/op" not in message
        # Positive half: a future over-eager strip that eats the whole message fails here.
        assert "identical lhs and rhs data types" in message
        assert "pl.tile operation 'matmul'" in message

    def test_cpp_check_detail_survives_on_the_cause(self):
        """The tail is hidden, not destroyed -- PTO_BACKTRACE=1 still reaches it."""
        with pytest.raises(InvalidOperationError) as exc_info:
            self._mismatched_matmul()

        cause = exc_info.value.__cause__
        assert isinstance(cause, ValueError)
        assert "Check failed" in str(cause)

    def test_rendered_header_is_followed_immediately_by_the_location_arrow(self):
        """Nothing may sit between the ``Error:`` header and the ``-->`` source arrow."""
        with pytest.raises(InvalidOperationError) as exc_info:
            self._mismatched_matmul()

        lines = ErrorRenderer(use_color=False).render(exc_info.value).split("\n")
        assert lines[0].startswith("Error:")
        assert lines[1].lstrip().startswith("-->")


class TestCheckSpanLocationStrippedFromHeader:
    """A ``CHECK_SPAN`` must not print its location inline in the bold ``Error:`` header.

    ``FatalLogger::~FatalLogger`` (``include/pypto/core/logging.h``) appends
    ``[<file>:<line>:<column>]`` *before* the newline that starts the ``Check failed:``
    tail, so the span's **absolute** path survives that tail's strip and lands in the
    header. The ``-->`` arrow below it already names a location -- and a different one,
    since the check's span is whatever IR node it was handed (here the ``tmp`` operand's
    definition) while the arrow is the call site.
    """

    @staticmethod
    def _mismatched_row_max():
        """Build a kernel whose only fault is an FP16 scratch tile for an FP32 reduction.

        Drives the CHECK_SPAN in src/ir/op/tile_ops/reduction.cpp, which passes
        ``args[1]->span_`` -- the ``tmp`` definition, one line above the failing call.
        """

        @pl.function
        def bad(t: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tile[[64, 1], pl.FP32]:
            a: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(t, offsets=[0, 0], shapes=[64, 64])
            tmp: pl.Tile[[64, 64], pl.FP16] = pl.tile.create([64, 64], pl.FP16)
            m: pl.Tile[[64, 1], pl.FP32] = pl.tile.row_max(a, tmp)
            return m

        return bad

    def test_header_carries_no_inline_source_location(self):
        """The op's message survives; the bracketed path it ended with does not."""
        with pytest.raises(InvalidOperationError) as exc_info:
            self._mismatched_row_max()

        message = exc_info.value.message
        assert "test_error_wrapping.py" not in message
        assert not message.rstrip().endswith("]")
        # Positive half: an over-eager strip that eats the payload fails here.
        assert "tmp_tile dtype fp16 and input dtype fp32" in message

    def test_rendered_diagnostic_still_locates_the_failing_call(self):
        """Dropping the inline span costs no location -- the arrow and snippet remain."""
        with pytest.raises(InvalidOperationError) as exc_info:
            self._mismatched_row_max()

        lines = ErrorRenderer(use_color=False).render(exc_info.value).split("\n")
        assert lines[0].startswith("Error:")
        assert "test_error_wrapping.py" not in lines[0]
        assert lines[1].lstrip().startswith("--> ")
        assert "test_error_wrapping.py" in lines[1]
        assert any("pl.tile.row_max(a, tmp)" in line for line in lines)

    def test_span_detail_survives_on_the_cause(self):
        """The location is hidden from the header, not destroyed."""
        with pytest.raises(InvalidOperationError) as exc_info:
            self._mismatched_row_max()

        cause = exc_info.value.__cause__
        assert isinstance(cause, ValueError)
        assert "test_error_wrapping.py" in str(cause)


class TestProgramCatchAll:
    """Tests that @pl.program wraps unexpected exceptions like @pl.function does."""

    def test_program_wraps_non_parser_error(self):
        """@pl.program wraps unexpected exceptions as ParserError subclass."""
        with pytest.raises(ParserError):

            @pl.program
            class BadProgram:
                @pl.function
                def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.BF16]:
                    result: pl.Tensor[[64], pl.BF16] = pl.tensor.cast(
                        x,
                        target_type=pl.BF16,
                        mode=99,  # type: ignore[arg-type]
                    )
                    return result

    def test_function_wraps_non_parser_error(self):
        """@pl.function wraps unexpected exceptions as ParserError subclass."""
        with pytest.raises(ParserError):

            @pl.function
            def bad_func(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.BF16]:
                result: pl.Tensor[[64], pl.BF16] = pl.tensor.cast(
                    x,
                    target_type=pl.BF16,
                    mode=99,  # type: ignore[arg-type]
                )
                return result

    def test_program_error_has_source_lines(self):
        """@pl.program attaches source lines to wrapped errors."""
        with pytest.raises(ParserError) as exc_info:

            @pl.program
            class SourceLinesProgram:
                @pl.function
                def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.BF16]:
                    result: pl.Tensor[[64], pl.BF16] = pl.tensor.cast(
                        x,
                        target_type=pl.BF16,
                        mode="bogus",  # type: ignore[arg-type]
                    )
                    return result

        assert exc_info.value.source_lines is not None


class TestTypeMismatchReassignment:
    """Tests for rejecting reassignment with a different type (#642)."""

    def test_reassign_same_type_succeeds(self):
        """Reassigning with the same type is allowed."""

        @pl.function
        def func(x: pl.Tensor[[16, 16], pl.FP32]) -> pl.Tensor[[16, 16], pl.FP32]:
            t = pl.create_tensor([16, 16], dtype=pl.FP32)
            t = pl.mul(t, 2.0)  # same type, should succeed
            return t

        # Both bindings survive, and the rebinding keeps the original type
        body = func.body
        assert isinstance(body, ir.SeqStmts)
        create_stmt, mul_stmt = (s for s in body.stmts if isinstance(s, ir.AssignStmt))
        assert isinstance(create_stmt.var.type, ir.TensorType)
        assert ir.structural_equal(mul_stmt.var.type, create_stmt.var.type)

    def test_reassign_different_shape_raises(self):
        """Reassigning with a different tensor shape raises ParserTypeError."""

        with pytest.raises(ParserTypeError, match="Cannot reassign"):

            @pl.function
            def func(  # noqa: F841
                x: pl.Tensor[[16, 16], pl.FP32],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                t = pl.create_tensor([16, 16], dtype=pl.FP32)  # noqa: F841
                t = pl.create_tensor([4, 4], dtype=pl.FP32)  # different shape  # noqa: F841
                return x


class TestBugClassErrorsAreNotWrapped:
    """Bug-class exceptions must escape the parser with type and traceback intact.

    The parser wraps stray exceptions as user-facing ParserErrors so a bad kernel gets
    a source-located diagnostic. A failed internal invariant is not a bad kernel - it is
    a PyPTO bug, and wrapping it hides both the `InternalError` type and the C++ stack
    trace that diagnoses it. Every broad `except Exception` on the parse path therefore
    lets `BUG_CLASS_EXCEPTIONS` through first.
    """

    @staticmethod
    def _raise_internal(*args, **kwargs):
        raise pypto.InternalError("Internal error: synthetic pass bug")

    def test_internal_error_from_op_is_not_wrapped(self, monkeypatch):
        """`_dispatch_op` re-raises rather than producing an InvalidOperationError."""
        monkeypatch.setattr(_dsl_tensor, "cast", self._raise_internal)

        with pytest.raises(pypto.InternalError, match="synthetic pass bug"):

            @pl.function
            def bad_kernel(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.BF16]:
                result: pl.Tensor[[64], pl.BF16] = pl.tensor.cast(x, target_type=pl.BF16)
                return result

    def test_assertion_error_from_op_is_not_wrapped(self, monkeypatch):
        """The other arm of BUG_CLASS_EXCEPTIONS.

        A bare `AssertionError` is bug-class for the same reason `InternalError` is, and
        pytest's own machinery depends on it propagating - wrapping one would turn a
        failed assertion anywhere under the parser into a confusing kernel diagnostic.
        """

        def _raise_assertion(*args, **kwargs):
            raise AssertionError("synthetic failed assertion")

        monkeypatch.setattr(_dsl_tensor, "cast", _raise_assertion)

        with pytest.raises(AssertionError, match="synthetic failed assertion") as exc_info:

            @pl.function
            def bad_kernel(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.BF16]:
                result: pl.Tensor[[64], pl.BF16] = pl.tensor.cast(x, target_type=pl.BF16)
                return result

        assert not isinstance(exc_info.value, ParserError)

    def test_internal_error_is_not_a_parser_error(self, monkeypatch):
        """Guards the specific regression: it must not be catchable as ParserError."""
        monkeypatch.setattr(_dsl_tensor, "cast", self._raise_internal)

        with pytest.raises(pypto.InternalError) as exc_info:

            @pl.function
            def bad_kernel(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.BF16]:
                result: pl.Tensor[[64], pl.BF16] = pl.tensor.cast(x, target_type=pl.BF16)
                return result

        assert not isinstance(exc_info.value, ParserError)

    def test_internal_error_survives_program_parsing(self, monkeypatch):
        """The @pl.program wrapper passes it through too, not just @pl.function."""
        monkeypatch.setattr(_dsl_tensor, "cast", self._raise_internal)

        with pytest.raises(pypto.InternalError, match="synthetic pass bug"):

            @pl.program
            class BadProgram:
                @pl.function
                def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.BF16]:
                    result: pl.Tensor[[64], pl.BF16] = pl.tensor.cast(x, target_type=pl.BF16)
                    return result

    def test_internal_error_from_closure_eval_is_not_wrapped(self, monkeypatch):
        """`ExprEvaluator.eval_expr` re-raises instead of producing a ParserTypeError.

        A compile-time closure expression can call into PyPTO and trip an internal
        invariant; wrapping that as "Failed to evaluate expression" erases both the type
        and the trace.
        """

        class _Boom:
            @property
            def value(self):
                raise pypto.InternalError("Internal error: synthetic pass bug")

        boom = _Boom()

        with pytest.raises(pypto.InternalError, match="synthetic pass bug"):

            @pl.function
            def bad_kernel(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                result: pl.Tensor[[64], pl.FP32] = pl.tensor.adds(x, boom.value)
                return result

    def test_ordinary_eval_failure_is_still_wrapped(self):
        """The passthrough must not leak ordinary eval failures past the diagnostics."""

        class _BadValue:
            @property
            def value(self):
                raise ValueError("not a compile-time constant")

        bad = _BadValue()

        with pytest.raises(ParserError):

            @pl.function
            def bad_kernel(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                result: pl.Tensor[[64], pl.FP32] = pl.tensor.adds(x, bad.value)
                return result

    def test_user_errors_are_still_wrapped(self, monkeypatch):
        """The passthrough must not leak ordinary user errors past the diagnostic layer."""

        def _raise_value(*args, **kwargs):
            raise ValueError("Invalid rounding mode 99")

        monkeypatch.setattr(_dsl_tensor, "cast", _raise_value)

        with pytest.raises(InvalidOperationError, match="Invalid rounding mode"):

            @pl.function
            def bad_kernel(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.BF16]:
                result: pl.Tensor[[64], pl.BF16] = pl.tensor.cast(x, target_type=pl.BF16)
                return result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
