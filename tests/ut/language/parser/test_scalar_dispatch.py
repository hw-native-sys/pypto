# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for scalar operation dispatch in the DSL parser.

Verifies that pl.min, pl.max, pl.cast dispatch to scalar IR ops
when called with scalar arguments.
"""

import pypto.language as pl
import pytest
from pypto.language.parser.diagnostics.exceptions import InvalidOperationError
from pypto.pypto_core import ir


class TestScalarMin:
    """Tests for pl.min dispatching to scalar ir.min_."""

    def test_scalar_min(self):
        """Test pl.min(scalar, scalar) prints and roundtrips correctly."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                config: pl.Tensor[[2], pl.INT64],
                out: pl.Tensor[[2, 16, 128], pl.FP32],
            ) -> pl.Tensor[[2, 16, 128], pl.FP32]:
                a: pl.Scalar[pl.INT64] = pl.tensor.read(config, [0])
                b: pl.Scalar[pl.INT64] = pl.tensor.read(config, [1])
                c: pl.Scalar[pl.INT64] = pl.min(a, b)
                _ = c + 1
                return out

        assert isinstance(Before, ir.Program)
        printed = Before.as_python()
        assert "pl.min(a, b)" in printed
        ir.assert_structural_equal(Before, pl.parse_program(printed))

    def test_scalar_min_with_literal(self):
        """Test pl.min(scalar, int_literal) prints and roundtrips correctly."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                config: pl.Tensor[[2], pl.INT64],
                out: pl.Tensor[[2, 16, 128], pl.FP32],
            ) -> pl.Tensor[[2, 16, 128], pl.FP32]:
                a: pl.Scalar[pl.INT64] = pl.tensor.read(config, [0])
                c: pl.Scalar[pl.INT64] = pl.min(a, 128)
                _ = c + 1
                return out

        assert isinstance(Before, ir.Program)
        printed = Before.as_python()
        assert "pl.min(a, 128)" in printed
        ir.assert_structural_equal(Before, pl.parse_program(printed))


class TestScalarMax:
    """Tests for pl.max dispatching to scalar ir.max_."""

    def test_scalar_max(self):
        """Test pl.max(scalar, scalar) prints and roundtrips correctly."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                config: pl.Tensor[[2], pl.INT64],
                out: pl.Tensor[[2, 16, 128], pl.FP32],
            ) -> pl.Tensor[[2, 16, 128], pl.FP32]:
                a: pl.Scalar[pl.INT64] = pl.tensor.read(config, [0])
                b: pl.Scalar[pl.INT64] = pl.tensor.read(config, [1])
                c: pl.Scalar[pl.INT64] = pl.max(a, b)
                _ = c + 1
                return out

        assert isinstance(Before, ir.Program)
        printed = Before.as_python()
        assert "pl.max(a, b)" in printed
        ir.assert_structural_equal(Before, pl.parse_program(printed))


class TestScalarCast:
    """Tests for pl.cast dispatching to scalar ir.cast."""

    def test_scalar_cast(self):
        """Test pl.cast(scalar, dtype) prints and roundtrips correctly."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                config: pl.Tensor[[2], pl.INT32],
                out: pl.Tensor[[2, 16, 128], pl.FP32],
            ) -> pl.Tensor[[2, 16, 128], pl.FP32]:
                a: pl.Scalar[pl.INT32] = pl.tensor.read(config, [0])
                b: pl.Scalar[pl.INDEX] = pl.cast(a, pl.INDEX)
                _ = b + 1
                return out

        assert isinstance(Before, ir.Program)
        printed = Before.as_python()
        assert "pl.cast(a, pl.INDEX)" in printed
        ir.assert_structural_equal(Before, pl.parse_program(printed))

    def test_scalar_cast_multiple_dtypes(self):
        """Test pl.cast(scalar, dtype) with different target dtypes."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                config: pl.Tensor[[2], pl.INT32],
                out: pl.Tensor[[2, 16, 128], pl.FP32],
            ) -> pl.Tensor[[2, 16, 128], pl.FP32]:
                a: pl.Scalar[pl.INT32] = pl.tensor.read(config, [0])
                b: pl.Scalar[pl.INDEX] = pl.cast(a, pl.INDEX)
                c: pl.Scalar[pl.INT64] = pl.cast(a, pl.INT64)
                _ = b + c
                return out

        assert isinstance(Before, ir.Program)
        printed = Before.as_python()
        assert "pl.cast(a, pl.INDEX)" in printed
        assert "pl.cast(a, pl.INT64)" in printed
        ir.assert_structural_equal(Before, pl.parse_program(printed))


class TestTileDispatchUnaffected:
    """Ensure tile ops still dispatch correctly when scalar dispatch is active."""

    def test_tile_min_still_works(self):
        """Ensure pl.min(tile, axis=...) still works as tile reduction."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                x: pl.Tensor[[32, 32], pl.FP32],
            ) -> pl.Tensor[[32, 32], pl.FP32]:
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(x, [0, 0], [32, 32])
                tile_c: pl.Tile[[32], pl.FP32] = pl.min(tile_a, axis=0)
                out: pl.Tensor[[32, 32], pl.FP32] = pl.store(tile_c, [0, 0], x)
                return out

        assert isinstance(Before, ir.Program)


class TestScalarNot:
    """Tests for `not` and `~` unary operator dispatch."""

    def test_logical_not_produces_not_node(self):
        """Test `not a` in DSL produces ir.Not and roundtrips correctly."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                config: pl.Tensor[[2], pl.INT64],
                out: pl.Tensor[[2, 16, 128], pl.FP32],
            ) -> pl.Tensor[[2, 16, 128], pl.FP32]:
                a: pl.Scalar[pl.INT64] = pl.tensor.read(config, [0])
                b: pl.Scalar[pl.BOOL] = not a
                _ = b
                return out

        assert isinstance(Before, ir.Program)
        func = list(Before.functions.values())[0]
        assert isinstance(func.body, ir.SeqStmts)
        let_stmt = func.body.stmts[1]
        assert isinstance(let_stmt, ir.AssignStmt)
        assert isinstance(let_stmt.value, ir.Not)
        printed = Before.as_python()
        assert "not a" in printed
        ir.assert_structural_equal(Before, pl.parse_program(printed))

    def test_bitwise_not_produces_bitnot_node(self):
        """Test `~a` in DSL produces ir.BitNot and roundtrips correctly."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                config: pl.Tensor[[2], pl.INT64],
                out: pl.Tensor[[2, 16, 128], pl.FP32],
            ) -> pl.Tensor[[2, 16, 128], pl.FP32]:
                a: pl.Scalar[pl.INT64] = pl.tensor.read(config, [0])
                b: pl.Scalar[pl.INT64] = ~a
                _ = b + 1
                return out

        assert isinstance(Before, ir.Program)
        func = list(Before.functions.values())[0]
        assert isinstance(func.body, ir.SeqStmts)
        let_stmt = func.body.stmts[1]
        assert isinstance(let_stmt, ir.AssignStmt)
        assert isinstance(let_stmt.value, ir.BitNot)
        printed = Before.as_python()
        assert "~a" in printed
        ir.assert_structural_equal(Before, pl.parse_program(printed))


class TestScalarArithmetic:
    """Tests for pl.add/sub/mul/div dispatching on scalar arguments."""

    def test_scalar_add(self):
        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                config: pl.Tensor[[2], pl.INT64],
                out: pl.Tensor[[2, 16, 128], pl.FP32],
            ) -> pl.Tensor[[2, 16, 128], pl.FP32]:
                a: pl.Scalar[pl.INT64] = pl.tensor.read(config, [0])
                b: pl.Scalar[pl.INT64] = pl.tensor.read(config, [1])
                c: pl.Scalar[pl.INT64] = pl.add(a, b)
                _ = c + 1
                return out

        assert isinstance(Before, ir.Program)
        printed = Before.as_python()
        assert "a + b" in printed
        ir.assert_structural_equal(Before, pl.parse_program(printed))

    def test_scalar_sub(self):
        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                config: pl.Tensor[[2], pl.INT64],
                out: pl.Tensor[[2, 16, 128], pl.FP32],
            ) -> pl.Tensor[[2, 16, 128], pl.FP32]:
                a: pl.Scalar[pl.INT64] = pl.tensor.read(config, [0])
                b: pl.Scalar[pl.INT64] = pl.tensor.read(config, [1])
                c: pl.Scalar[pl.INT64] = pl.sub(a, b)
                _ = c + 1
                return out

        assert isinstance(Before, ir.Program)
        printed = Before.as_python()
        assert "a - b" in printed
        ir.assert_structural_equal(Before, pl.parse_program(printed))

    def test_scalar_mul(self):
        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                config: pl.Tensor[[2], pl.INT64],
                out: pl.Tensor[[2, 16, 128], pl.FP32],
            ) -> pl.Tensor[[2, 16, 128], pl.FP32]:
                a: pl.Scalar[pl.INT64] = pl.tensor.read(config, [0])
                b: pl.Scalar[pl.INT64] = pl.tensor.read(config, [1])
                c: pl.Scalar[pl.INT64] = pl.mul(a, b)
                _ = c + 1
                return out

        assert isinstance(Before, ir.Program)
        printed = Before.as_python()
        assert "a * b" in printed
        ir.assert_structural_equal(Before, pl.parse_program(printed))

    def test_scalar_div(self):
        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                config: pl.Tensor[[2], pl.FP32],
                out: pl.Tensor[[2, 16, 128], pl.FP32],
            ) -> pl.Tensor[[2, 16, 128], pl.FP32]:
                a: pl.Scalar[pl.FP32] = pl.tensor.read(config, [0])
                b: pl.Scalar[pl.FP32] = pl.tensor.read(config, [1])
                c: pl.Scalar[pl.FP32] = pl.div(a, b)
                _ = c
                return out

        assert isinstance(Before, ir.Program)
        printed = Before.as_python()
        assert "a / b" in printed
        ir.assert_structural_equal(Before, pl.parse_program(printed))

    def test_scalar_add_matches_plus_operator(self):
        """`pl.add(a, b)` must produce the same IR as `a + b`."""

        @pl.program
        class WithCall:
            @pl.function
            def main(
                self,
                config: pl.Tensor[[2], pl.INT64],
                out: pl.Tensor[[2, 16, 128], pl.FP32],
            ) -> pl.Tensor[[2, 16, 128], pl.FP32]:
                a: pl.Scalar[pl.INT64] = pl.tensor.read(config, [0])
                b: pl.Scalar[pl.INT64] = pl.tensor.read(config, [1])
                c: pl.Scalar[pl.INT64] = pl.add(a, b)
                _ = c
                return out

        @pl.program
        class WithOperator:
            @pl.function
            def main(
                self,
                config: pl.Tensor[[2], pl.INT64],
                out: pl.Tensor[[2, 16, 128], pl.FP32],
            ) -> pl.Tensor[[2, 16, 128], pl.FP32]:
                a: pl.Scalar[pl.INT64] = pl.tensor.read(config, [0])
                b: pl.Scalar[pl.INT64] = pl.tensor.read(config, [1])
                c: pl.Scalar[pl.INT64] = a + b
                _ = c
                return out

        ir.assert_structural_equal(WithCall, WithOperator)


class TestScalarUnsupportedOpHint:
    """Verify the catch-all error message points users at Python operators."""

    def test_unsupported_op_hint_mentions_python_operators(self):
        # `pl.exp` is a unified op (tile/tensor) with no scalar dispatch — hits
        # the catch-all path that surfaces the improved hint.
        with pytest.raises(InvalidOperationError, match=r"pl\.exp.*expected.*got Scalar") as exc_info:

            @pl.program
            class Bad:
                @pl.function
                def main(
                    self,
                    config: pl.Tensor[[1], pl.FP32],
                    out: pl.Tensor[[2, 16, 128], pl.FP32],
                ) -> pl.Tensor[[2, 16, 128], pl.FP32]:
                    a: pl.Scalar[pl.FP32] = pl.tensor.read(config, [0])
                    _ = pl.exp(a)  # pyright: ignore[reportArgumentType]
                    return out

            assert Bad is not None  # silence "unused" warnings if reached

        err = exc_info.value
        assert "exp" in err.message  # type: ignore[attr-defined]
        assert "Python operators" in err.hint  # type: ignore[attr-defined]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
