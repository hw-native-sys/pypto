# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for parser error handling."""

import pypto
import pypto.language as pl
import pytest
from pypto.language.parser.diagnostics import (
    InvalidOperationError,
    ParserSyntaxError,
    ParserTypeError,
    SSAViolationError,
    UndefinedVariableError,
    UnsupportedFeatureError,
)
from pypto.language.parser.diagnostics.renderer import ErrorRenderer


class TestErrorCases:
    """Tests for parser error handling and validation."""

    def test_missing_parameter_annotation(self):
        """Test error when parameter lacks type annotation."""

        with pytest.raises(ParserTypeError, match="missing type annotation"):

            @pl.function
            def no_annotation(x):
                return x

    def test_missing_return_annotation(self):
        """Test function without return annotation still works."""

        @pl.function
        def no_return_type(x: pl.Tensor[[64], pl.FP32]):
            result: pl.Tensor[[64], pl.FP32] = pl.mul(x, 2.0)
            return result

        # Should still parse successfully
        assert isinstance(no_return_type, pypto.ir.Function)

    def test_undefined_variable_reference(self):
        """Test error when referencing undefined variable."""

        with pytest.raises(UndefinedVariableError, match="Undefined variable"):

            @pl.function
            def undefined_var(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                result: pl.Tensor[[64], pl.FP32] = pl.add(x, undefined)  # noqa: F821 # type: ignore
                return result

    def test_invalid_tensor_type_syntax(self):
        """Test error on invalid tensor type syntax."""

        with pytest.raises(ParserTypeError):

            @pl.function
            def invalid_type(x: pl.Tensor) -> pl.Tensor[[64], pl.FP32]:  # type: ignore
                return x

    def test_iter_arg_init_values_mismatch(self):
        """Test error when iter_args don't match init_values count."""

        with pytest.raises(ParserSyntaxError, match="Mismatch"):

            @pl.function
            def mismatch(n: pl.Tensor[[1], pl.INT32]) -> pl.Tensor[[1], pl.INT32]:
                init1: pl.Tensor[[1], pl.INT32] = pl.create_tensor([1], dtype=pl.INT32)
                init2: pl.Tensor[[1], pl.INT32] = pl.create_tensor([1], dtype=pl.INT32)

                # 3 iter_args but only 2 init_values
                for i, (v1, v2, v3) in pl.range(5, init_values=(init1, init2)):  # type: ignore
                    out1, out2, out3 = pl.yield_(v1, v2, v3)

                return out1

    def test_unsupported_statement_type(self):
        """Test error on unsupported Python statement."""

        with pytest.raises(UnsupportedFeatureError, match="Unsupported"):

            @pl.function
            def unsupported(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                # Try/except is not supported
                try:
                    result: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
                except:  # noqa: E722
                    result: pl.Tensor[[64], pl.FP32] = x
                return result

    def test_invalid_range_usage(self):
        """Test error when for loop doesn't use pl.range()."""

        with pytest.raises(ParserSyntaxError, match=r"must use pl\.range\(\)"):

            @pl.function
            def invalid_loop(n: pl.Tensor[[1], pl.INT32]) -> pl.Tensor[[1], pl.INT32]:
                result: pl.Tensor[[1], pl.INT32] = n
                # Using Python range() instead of pl.range()
                for i in range(10):
                    result = pl.add(result, 1)
                return result

    def test_invalid_loop_target_format(self):
        """Test error on invalid for loop target format."""

        with pytest.raises(ParserSyntaxError, match="target must be"):

            @pl.function
            def bad_target(n: pl.Tensor[[1], pl.INT32]) -> pl.Tensor[[1], pl.INT32]:
                init: pl.Tensor[[1], pl.INT32] = pl.create_tensor([1], dtype=pl.INT32)

                # Missing iter_args tuple
                for i in pl.range(5, init_values=(init,)):
                    result = pl.yield_(i)  # type: ignore

                return result

    def test_chunked_loop_requires_auto_incore(self):
        """Test that chunked loops are rejected outside auto_incore scope."""
        code = """
import pypto.language as pl

@pl.program
class ChunkedLoopProgram:
    @pl.function(type=pl.FunctionType.Orchestration)
    def main(
        self,
        x: pl.Tensor[[16, 4], pl.FP32],
        seq_lens: pl.Tensor[[16], pl.INT32],
    ) -> pl.Tensor[[16, 4], pl.FP32]:
        for b in pl.parallel(0, 16, 1, chunk=4, chunk_policy="leading_full"):
            _ctx_len = pl.tensor.read(seq_lens, [b])
        return x
"""
        with pytest.raises(
            ParserSyntaxError,
            match=r"chunk=\.\.\. loops are only valid inside with pl\.at",
        ):
            pl.parse_program(code)

    def test_unknown_tensor_operation(self):
        """Test error on unknown tensor operation."""

        with pytest.raises(InvalidOperationError, match="Unknown tensor operation"):

            @pl.function
            def unknown_op(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                # nonexistent_op doesn't exist
                result: pl.Tensor[[64], pl.FP32] = pl.nonexistent_op(x)  # type: ignore
                return result


class TestSSAValidation:
    """Tests for SSA validation."""

    def test_ssa_violation_double_assignment(self):
        """Test that double assignment in same scope is caught with strict_ssa=True."""

        with pytest.raises(SSAViolationError):

            @pl.function(strict_ssa=True)
            def double_assign(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                # First assignment
                result: pl.Tensor[[64], pl.FP32] = pl.mul(x, 2.0)
                # Second assignment (SSA violation)
                result: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
                return result

    def test_variable_from_inner_scope_not_accessible(self):
        """Test that variables from inner scopes aren't accessible without yield."""

        # Note: This test demonstrates the expected behavior
        # The current implementation tracks yields, so this should work correctly
        @pl.function
        def scope_test(n: pl.Tensor[[1], pl.INT32]) -> pl.Tensor[[64], pl.FP32]:
            init: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)

            for i, (acc,) in pl.range(5, init_values=(init,)):
                temp: pl.Tensor[[64], pl.FP32] = pl.add(acc, 1.0)
                # temp is yielded, so it's accessible as 'result'
                result = pl.yield_(temp)

            # Can return result because it was yielded from loop
            return result

        assert isinstance(scope_test, pypto.ir.Function)


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_function_body(self):
        """Test function with minimal body (just return)."""

        @pl.function
        def minimal(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            return x

        assert isinstance(minimal, pypto.ir.Function)

    def test_single_dimension_tensor(self):
        """Test tensor with single dimension."""

        @pl.function
        def single_dim(x: pl.Tensor[[128], pl.FP32]) -> pl.Tensor[[128], pl.FP32]:
            result: pl.Tensor[[128], pl.FP32] = pl.mul(x, 2.0)
            return result

        assert isinstance(single_dim, pypto.ir.Function)

    def test_three_dimension_tensor(self):
        """Test tensor with three dimensions."""

        @pl.function
        def three_dim(
            x: pl.Tensor[[64, 128, 256], pl.FP32],
        ) -> pl.Tensor[[64, 128, 256], pl.FP32]:
            return x

        assert isinstance(three_dim, pypto.ir.Function)

    def test_loop_with_range_one(self):
        """Test loop that executes only once."""

        @pl.function
        def one_iteration(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            for i, (acc,) in pl.range(1, init_values=(x,)):
                result = pl.yield_(acc)

            return result

        assert isinstance(one_iteration, pypto.ir.Function)

    def test_loop_with_start_stop_step(self):
        """Test loop with start, stop, and step parameters."""

        @pl.function
        def custom_range(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            for i, (acc,) in pl.range(2, 10, 2, init_values=(x,)):
                new_acc: pl.Tensor[[64], pl.FP32] = pl.add(acc, 1.0)
                result = pl.yield_(new_acc)

            return result

        assert isinstance(custom_range, pypto.ir.Function)

    def test_function_with_many_variables(self):
        """Test function with many local variables."""

        @pl.function
        def many_vars(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            v1: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
            v2: pl.Tensor[[64], pl.FP32] = pl.add(v1, 2.0)
            v3: pl.Tensor[[64], pl.FP32] = pl.add(v2, 3.0)
            v4: pl.Tensor[[64], pl.FP32] = pl.add(v3, 4.0)
            v5: pl.Tensor[[64], pl.FP32] = pl.add(v4, 5.0)
            v6: pl.Tensor[[64], pl.FP32] = pl.add(v5, 6.0)
            v7: pl.Tensor[[64], pl.FP32] = pl.add(v6, 7.0)
            v8: pl.Tensor[[64], pl.FP32] = pl.add(v7, 8.0)
            v9: pl.Tensor[[64], pl.FP32] = pl.add(v8, 9.0)
            v10: pl.Tensor[[64], pl.FP32] = pl.add(v9, 10.0)
            return v10

        assert isinstance(many_vars, pypto.ir.Function)

    def test_different_shape_tensors(self):
        """Test function with tensors of different shapes."""

        @pl.function
        def diff_shapes(
            a: pl.Tensor[[64], pl.FP32],
            b: pl.Tensor[[128], pl.FP32],
            c: pl.Tensor[[256], pl.FP32],
        ) -> pl.Tensor[[64], pl.FP32]:
            return a

        assert isinstance(diff_shapes, pypto.ir.Function)
        assert len(diff_shapes.params) == 3


class TestAnnotationConsistency:
    """Integration tests for annotation vs inferred type checking."""

    def test_annotation_shape_mismatch_on_load(self):
        """Annotation says [128] but pl.load infers [64] — must raise."""
        with pytest.raises(ParserTypeError, match="shape dimension 0 = 128.*64"):

            @pl.function
            def bad_shape(x: pl.Tensor[[128], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                _x_tile: pl.Tile[[128], pl.FP32] = pl.tile.load(x, [0], [64])
                return x

    def test_annotation_dtype_mismatch_on_load(self):
        """Annotation says FP16 but tile.load infers FP32 — must raise."""
        with pytest.raises(ParserTypeError, match="dtype"):

            @pl.function
            def bad_dtype(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                _result: pl.Tile[[64], pl.FP16] = pl.tile.load(x, [0], [64])
                return x

    def test_annotation_matches_inferred_type(self):
        """Correct annotation passes (regression test)."""

        @pl.function
        def correct(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            x_tile: pl.Tile[[64], pl.FP32] = pl.tile.load(x, [0], [64])
            result: pl.Tensor[[64], pl.FP32] = pl.tile.store(x_tile, [0], output_tensor=x)
            return result

        assert isinstance(correct, pypto.ir.Function)


class TestScalarTypeErrors:
    """Tests for Scalar type error handling."""

    def test_scalar_without_dtype(self):
        """Test that Scalar without dtype raises error."""
        with pytest.raises(pl.parser.ParserError):

            @pl.function
            def bad_scalar(x: pl.Scalar) -> pl.Scalar:  # Missing [dtype]
                return x


class TestConditionMustBeBool:
    """Tests for if/while conditions requiring BOOL dtype (RFC #993)."""

    def test_if_with_int_literal_rejected(self):
        """Test that `if 1:` is rejected — integer literal is not BOOL."""
        with pytest.raises(ParserTypeError, match="if condition must be a Bool-typed scalar"):

            @pl.function
            def bad_if(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                if 1:
                    result: pl.Tensor[[64], pl.FP32] = pl.mul(x, 2.0)
                else:
                    result: pl.Tensor[[64], pl.FP32] = x
                return result

    def test_while_with_int_literal_rejected(self):
        """Test that `while 1:` is rejected — integer literal is not BOOL."""
        with pytest.raises(ParserTypeError, match="while condition must be a Bool-typed scalar"):

            @pl.function
            def bad_while(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                while 1:
                    x = pl.mul(x, 2.0)
                return x

    def test_if_with_bool_literal_accepted(self):
        """Test that `if True:` is accepted."""

        @pl.function
        def ok_if(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            if True:
                result: pl.Tensor[[64], pl.FP32] = pl.mul(x, 2.0)
            else:
                result: pl.Tensor[[64], pl.FP32] = x
            return result

        assert isinstance(ok_if, pypto.ir.Function)

    def test_if_with_comparison_accepted(self):
        """Test that comparison condition (BOOL-typed) is accepted."""

        @pl.function
        def ok_cmp(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            init: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
            for i, (acc,) in pl.range(5, init_values=(init,)):
                if i > 0:
                    new_val: pl.Tensor[[64], pl.FP32] = pl.mul(acc, 2.0)
                    val: pl.Tensor[[64], pl.FP32] = pl.yield_(new_val)
                else:
                    val: pl.Tensor[[64], pl.FP32] = pl.yield_(acc)
                result = pl.yield_(val)
            return result

        assert isinstance(ok_cmp, pypto.ir.Function)

    def test_pl_cond_with_int_literal_rejected(self):
        """Test that `pl.cond(1)` inside `pl.while_()` is rejected (same check as natural while)."""
        with pytest.raises(ParserTypeError, match="while condition must be a Bool-typed scalar"):

            @pl.function
            def bad_pl_while(n: pl.Scalar[pl.INT64]) -> pl.Scalar[pl.INT64]:
                x: pl.Scalar[pl.INT64] = 0
                for (x_iter,) in pl.while_(init_values=(x,)):
                    pl.cond(1)  # type: ignore  # non-bool
                    y = pl.yield_(x_iter + 1)
                return y  # noqa: F821


class TestYieldLHSRequired:
    """`pl.range(init_values=...)` and `pl.while_(init_values=...)` both
    require an assignment-form `pl.yield_(...)` so the LHS supplies the
    post-loop binding name (uniform convention across both loop forms)."""

    def test_pl_range_bare_yield_with_init_values_rejected(self):
        with pytest.raises(ParserSyntaxError, match=r"requires an assignment-form pl\.yield_"):

            @pl.function
            def bad_range(x_0: pl.Scalar[pl.INT64]) -> pl.Scalar[pl.INT64]:
                for _i, (x_iter,) in pl.range(10, init_values=(x_0,)):
                    pl.yield_(x_iter + 1)  # bare — no LHS
                return x_iter  # noqa: F821

    def test_pl_while_bare_yield_with_init_values_rejected(self):
        with pytest.raises(ParserSyntaxError, match=r"requires an assignment-form pl\.yield_"):

            @pl.function
            def bad_while(n: pl.Scalar[pl.INT64]) -> pl.Scalar[pl.INT64]:
                x: pl.Scalar[pl.INT64] = 0
                for (x_iter,) in pl.while_(init_values=(x,)):
                    pl.cond(x_iter < n)
                    pl.yield_(x_iter + 1)  # bare — no LHS
                return x  # noqa: F821


class TestSourceLocationPreservation:
    """Regression tests for issue #1200: every frontend diagnostic must
    include the user's source location so the error can be navigated to."""

    @staticmethod
    def _assert_span_populated(err: pl.parser.ParserError) -> None:
        span = err.span
        assert span is not None, "ParserError must carry a span for location preservation"
        assert span["filename"], f"span filename must be non-empty, got {span['filename']!r}"
        assert span["begin_line"] > 0, f"span begin_line must be > 0, got {span['begin_line']}"
        assert span["begin_column"] >= 0, f"span begin_column must be >= 0, got {span['begin_column']}"

    def test_unsupported_type_annotation_carries_span(self):
        """The exact failure mode reported in issue #1200."""
        with pytest.raises(ParserTypeError) as excinfo:

            @pl.function
            def bad(x: ([64, 128], pl.FP32)) -> pl.Tensor[[64], pl.FP32]:  # type: ignore
                return x  # type: ignore

        self._assert_span_populated(excinfo.value)

    def test_incomplete_type_annotation_carries_span(self):
        """Bare `pl.Tensor` (no subscript) — must report location."""
        with pytest.raises(ParserTypeError) as excinfo:

            @pl.function
            def bad(x: pl.Tensor) -> pl.Tensor[[64], pl.FP32]:  # type: ignore
                return x

        self._assert_span_populated(excinfo.value)

    def test_parameter_tuple_annotation_carries_span(self):
        """A tuple-as-parameter-type annotation must be located.

        ``resolve_param_type`` rejects list-typed (tuple) results because
        parameters cannot be tuples. The error must include source location.
        """
        with pytest.raises(ParserTypeError) as excinfo:

            @pl.function
            def bad(  # type: ignore
                x: tuple[pl.Tensor[[64], pl.FP32], pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                return x  # type: ignore

        self._assert_span_populated(excinfo.value)

    def test_rendered_error_includes_location_arrow(self):
        """End-to-end pipeline (raise → attach source_lines → render) emits
        the `--> file:line:col` arrow that issue #1200 reported as missing."""
        with pytest.raises(ParserTypeError) as excinfo:

            @pl.function
            def bad(x: ([64, 128], pl.FP32)) -> pl.Tensor[[64], pl.FP32]:  # type: ignore
                return x  # type: ignore

        err = excinfo.value
        # `@pl.function` attaches source_lines via its except handler before
        # the exception escapes, so the renderer has everything it needs.
        rendered = ErrorRenderer(use_color=False).render(err)
        assert "-->" in rendered, f"rendered output missing location arrow:\n{rendered}"
        # Must be a real line:col, not the unknown-span default of `:0:0`.
        assert ":0:0" not in rendered, f"rendered output has unknown span:\n{rendered}"

    def test_pl_at_unknown_kwarg_carries_span(self):
        """Covers the ast_parser.py path: ParserSyntaxError from pl.at()
        keyword dispatch must include the source location of the bad kwarg."""
        code = """
import pypto.language as pl

@pl.program
class P:
    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, x: pl.Tensor[[16], pl.FP32]) -> pl.Tensor[[16], pl.FP32]:
        with pl.at(level=pl.Level.HOST, not_a_real_arg=42):
            pass
        return x
"""
        with pytest.raises(ParserSyntaxError) as excinfo:
            pl.parse_program(code)

        self._assert_span_populated(excinfo.value)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
