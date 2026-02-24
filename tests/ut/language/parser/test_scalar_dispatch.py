# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for scalar operation dispatch in the DSL parser.

Verifies that pl.min, pl.max dispatch to scalar IR ops
when called with scalar arguments.
"""

import pypto.language as pl
import pytest
from pypto.pypto_core import ir


class TestScalarOpDispatch:
    """Tests for scalar operation dispatch through pl.* interface."""

    def test_scalar_min(self):
        """Test pl.min(scalar, scalar) dispatches to ir.min_."""

        @pl.function(type=pl.FunctionType.Orchestration)
        def test_min(
            config: pl.Tensor[[2], pl.INT64],
            out: pl.Tensor[[2, 16, 128], pl.FP32],
        ) -> pl.Tensor[[2, 16, 128], pl.FP32]:
            a: pl.Scalar[pl.UINT64] = pl.tensor.read(config, [0])
            b: pl.Scalar[pl.UINT64] = pl.tensor.read(config, [1])
            c = pl.min(a, b)
            _ = c + 1
            return out

        assert isinstance(test_min, ir.Function)
        ir_text = ir.python_print(test_min)
        assert "min" in ir_text.lower()

    def test_scalar_max(self):
        """Test pl.max(scalar, scalar) dispatches to ir.max_."""

        @pl.function(type=pl.FunctionType.Orchestration)
        def test_max(
            config: pl.Tensor[[2], pl.INT64],
            out: pl.Tensor[[2, 16, 128], pl.FP32],
        ) -> pl.Tensor[[2, 16, 128], pl.FP32]:
            a: pl.Scalar[pl.UINT64] = pl.tensor.read(config, [0])
            b: pl.Scalar[pl.UINT64] = pl.tensor.read(config, [1])
            c = pl.max(a, b)
            _ = c + 1
            return out

        assert isinstance(test_max, ir.Function)
        ir_text = ir.python_print(test_max)
        assert "max" in ir_text.lower()

    def test_scalar_min_with_literal(self):
        """Test pl.min(scalar, int_literal) — the paged_attention use case."""

        @pl.function(type=pl.FunctionType.Orchestration)
        def test_min_lit(
            config: pl.Tensor[[2], pl.INT64],
            out: pl.Tensor[[2, 16, 128], pl.FP32],
        ) -> pl.Tensor[[2, 16, 128], pl.FP32]:
            a: pl.Scalar[pl.UINT64] = pl.tensor.read(config, [0])
            c = pl.min(a, 128)
            _ = c + 1
            return out

        assert isinstance(test_min_lit, ir.Function)
        ir_text = ir.python_print(test_min_lit)
        assert "min" in ir_text.lower()

    def test_tile_min_still_works(self):
        """Ensure pl.min(tile, axis=...) still works as tile reduction."""

        @pl.function
        def test_tile_min(
            x: pl.Tensor[[32, 32], pl.FP32],
        ) -> pl.Tensor[[32, 32], pl.FP32]:
            tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(x, [0, 0], [32, 32])
            tile_c: pl.Tile[[1, 32], pl.FP32] = pl.min(tile_a, axis=0)
            out: pl.Tensor[[32, 32], pl.FP32] = pl.store(tile_c, [0, 0], [1, 32], x)
            return out

        assert isinstance(test_tile_min, ir.Function)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
