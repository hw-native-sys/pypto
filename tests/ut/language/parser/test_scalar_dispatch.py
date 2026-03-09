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
from pypto.pypto_core import ir


class TestScalarMin:
    """Tests for pl.min dispatching to scalar ir.min_."""

    def test_scalar_min(self):
        """Test pl.min(scalar, scalar) prints and roundtrips correctly."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Orchestration)
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
            @pl.function(type=pl.FunctionType.Orchestration)
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
            @pl.function(type=pl.FunctionType.Orchestration)
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
            @pl.function(type=pl.FunctionType.Orchestration)
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
            @pl.function(type=pl.FunctionType.Orchestration)
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
