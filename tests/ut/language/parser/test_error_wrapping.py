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

import pypto.language as pl
import pytest
from pypto.language.parser.diagnostics import (
    InvalidOperationError,
    ParserError,
    ParserTypeError,
)


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

        assert func is not None

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
