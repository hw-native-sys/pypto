# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for unified operation dispatch (pl.op.*).

Each test builds two functions — one using the unified ``pl.op.X`` API and one
using the explicit ``pl.op.tensor.X`` / ``pl.op.block.X`` API — then asserts
they produce structurally equal IR.
"""

import pypto.language as pl
import pytest
from pypto.language.op import unified_ops
from pypto.pypto_core import ir


class TestUnifiedTensorDispatch:
    """pl.op.X with Tensor args produces the same IR as pl.op.tensor.X."""

    def test_add(self):
        @pl.function
        def unified(a: pl.Tensor[[64], pl.FP32], b: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            c: pl.Tensor[[64], pl.FP32] = pl.op.add(a, b)
            return c

        @pl.function
        def explicit(a: pl.Tensor[[64], pl.FP32], b: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            c: pl.Tensor[[64], pl.FP32] = pl.op.tensor.add(a, b)
            return c

        ir.assert_structural_equal(unified, explicit)

    def test_sub(self):
        @pl.function
        def unified(a: pl.Tensor[[64], pl.FP32], b: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            c: pl.Tensor[[64], pl.FP32] = pl.op.sub(a, b)
            return c

        @pl.function
        def explicit(a: pl.Tensor[[64], pl.FP32], b: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            c: pl.Tensor[[64], pl.FP32] = pl.op.tensor.sub(a, b)
            return c

        ir.assert_structural_equal(unified, explicit)

    def test_mul(self):
        @pl.function
        def unified(a: pl.Tensor[[64], pl.FP32], b: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            c: pl.Tensor[[64], pl.FP32] = pl.op.mul(a, b)
            return c

        @pl.function
        def explicit(a: pl.Tensor[[64], pl.FP32], b: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            c: pl.Tensor[[64], pl.FP32] = pl.op.tensor.mul(a, b)
            return c

        ir.assert_structural_equal(unified, explicit)

    def test_div(self):
        @pl.function
        def unified(a: pl.Tensor[[64], pl.FP32], b: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            c: pl.Tensor[[64], pl.FP32] = pl.op.div(a, b)
            return c

        @pl.function
        def explicit(a: pl.Tensor[[64], pl.FP32], b: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            c: pl.Tensor[[64], pl.FP32] = pl.op.tensor.div(a, b)
            return c

        ir.assert_structural_equal(unified, explicit)

    def test_maximum(self):
        @pl.function
        def unified(a: pl.Tensor[[64], pl.FP32], b: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            c: pl.Tensor[[64], pl.FP32] = pl.op.maximum(a, b)
            return c

        @pl.function
        def explicit(a: pl.Tensor[[64], pl.FP32], b: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            c: pl.Tensor[[64], pl.FP32] = pl.op.tensor.maximum(a, b)
            return c

        ir.assert_structural_equal(unified, explicit)

    def test_exp(self):
        @pl.function
        def unified(a: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            c: pl.Tensor[[64], pl.FP32] = pl.op.exp(a)
            return c

        @pl.function
        def explicit(a: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            c: pl.Tensor[[64], pl.FP32] = pl.op.tensor.exp(a)
            return c

        ir.assert_structural_equal(unified, explicit)

    def test_add_scalar(self):
        @pl.function
        def unified(a: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            c: pl.Tensor[[64], pl.FP32] = pl.op.add(a, 5)
            return c

        @pl.function
        def explicit(a: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            c: pl.Tensor[[64], pl.FP32] = pl.op.tensor.add(a, 5)
            return c

        ir.assert_structural_equal(unified, explicit)

    def test_matmul(self):
        @pl.function
        def unified(
            a: pl.Tensor[[64, 128], pl.FP16], b: pl.Tensor[[128, 64], pl.FP16]
        ) -> pl.Tensor[[64, 64], pl.FP16]:
            c: pl.Tensor[[64, 64], pl.FP16] = pl.op.matmul(a, b)
            return c

        @pl.function
        def explicit(
            a: pl.Tensor[[64, 128], pl.FP16], b: pl.Tensor[[128, 64], pl.FP16]
        ) -> pl.Tensor[[64, 64], pl.FP16]:
            c: pl.Tensor[[64, 64], pl.FP16] = pl.op.tensor.matmul(a, b)
            return c

        ir.assert_structural_equal(unified, explicit)

    def test_row_max(self):
        @pl.function
        def unified(a: pl.Tensor[[64, 128], pl.FP32]) -> pl.Tensor[[64, 1], pl.FP32]:
            c: pl.Tensor[[64, 1], pl.FP32] = pl.op.row_max(a)
            return c

        @pl.function
        def explicit(a: pl.Tensor[[64, 128], pl.FP32]) -> pl.Tensor[[64, 1], pl.FP32]:
            c: pl.Tensor[[64, 1], pl.FP32] = pl.op.tensor.row_max(a)
            return c

        ir.assert_structural_equal(unified, explicit)

    def test_row_sum(self):
        @pl.function
        def unified(a: pl.Tensor[[64, 128], pl.FP32]) -> pl.Tensor[[64, 1], pl.FP32]:
            c: pl.Tensor[[64, 1], pl.FP32] = pl.op.row_sum(a)
            return c

        @pl.function
        def explicit(a: pl.Tensor[[64, 128], pl.FP32]) -> pl.Tensor[[64, 1], pl.FP32]:
            c: pl.Tensor[[64, 1], pl.FP32] = pl.op.tensor.row_sum(a)
            return c

        ir.assert_structural_equal(unified, explicit)

    def test_reshape(self):
        @pl.function
        def unified(a: pl.Tensor[[64, 128], pl.FP32]) -> pl.Tensor[[128, 64], pl.FP32]:
            c: pl.Tensor[[128, 64], pl.FP32] = pl.op.reshape(a, [128, 64])
            return c

        @pl.function
        def explicit(a: pl.Tensor[[64, 128], pl.FP32]) -> pl.Tensor[[128, 64], pl.FP32]:
            c: pl.Tensor[[128, 64], pl.FP32] = pl.op.tensor.reshape(a, [128, 64])
            return c

        ir.assert_structural_equal(unified, explicit)


class TestUnifiedBlockDispatch:
    """pl.op.X with Tile args produces the same IR as pl.op.block.X."""

    def test_add(self):
        @pl.function
        def unified(
            t: pl.Tensor[[64, 64], pl.FP32], out: pl.Tensor[[64, 64], pl.FP32]
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            a: pl.Tile[[64, 64], pl.FP32] = pl.op.block.load(t, 0, 0, 64, 64)
            b: pl.Tile[[64, 64], pl.FP32] = pl.op.block.load(t, 0, 0, 64, 64)
            c: pl.Tile[[64, 64], pl.FP32] = pl.op.add(a, b)
            result: pl.Tensor[[64, 64], pl.FP32] = pl.op.block.store(c, 0, 0, 64, 64, out)
            return result

        @pl.function
        def explicit(
            t: pl.Tensor[[64, 64], pl.FP32], out: pl.Tensor[[64, 64], pl.FP32]
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            a: pl.Tile[[64, 64], pl.FP32] = pl.op.block.load(t, 0, 0, 64, 64)
            b: pl.Tile[[64, 64], pl.FP32] = pl.op.block.load(t, 0, 0, 64, 64)
            c: pl.Tile[[64, 64], pl.FP32] = pl.op.block.add(a, b)
            result: pl.Tensor[[64, 64], pl.FP32] = pl.op.block.store(c, 0, 0, 64, 64, out)
            return result

        ir.assert_structural_equal(unified, explicit)

    def test_sub(self):
        @pl.function
        def unified(
            t: pl.Tensor[[64, 64], pl.FP32], out: pl.Tensor[[64, 64], pl.FP32]
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            a: pl.Tile[[64, 64], pl.FP32] = pl.op.block.load(t, 0, 0, 64, 64)
            b: pl.Tile[[64, 64], pl.FP32] = pl.op.block.load(t, 0, 0, 64, 64)
            c: pl.Tile[[64, 64], pl.FP32] = pl.op.sub(a, b)
            result: pl.Tensor[[64, 64], pl.FP32] = pl.op.block.store(c, 0, 0, 64, 64, out)
            return result

        @pl.function
        def explicit(
            t: pl.Tensor[[64, 64], pl.FP32], out: pl.Tensor[[64, 64], pl.FP32]
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            a: pl.Tile[[64, 64], pl.FP32] = pl.op.block.load(t, 0, 0, 64, 64)
            b: pl.Tile[[64, 64], pl.FP32] = pl.op.block.load(t, 0, 0, 64, 64)
            c: pl.Tile[[64, 64], pl.FP32] = pl.op.block.sub(a, b)
            result: pl.Tensor[[64, 64], pl.FP32] = pl.op.block.store(c, 0, 0, 64, 64, out)
            return result

        ir.assert_structural_equal(unified, explicit)

    def test_exp(self):
        @pl.function
        def unified(
            t: pl.Tensor[[64, 64], pl.FP32], out: pl.Tensor[[64, 64], pl.FP32]
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            a: pl.Tile[[64, 64], pl.FP32] = pl.op.block.load(t, 0, 0, 64, 64)
            b: pl.Tile[[64, 64], pl.FP32] = pl.op.exp(a)
            result: pl.Tensor[[64, 64], pl.FP32] = pl.op.block.store(b, 0, 0, 64, 64, out)
            return result

        @pl.function
        def explicit(
            t: pl.Tensor[[64, 64], pl.FP32], out: pl.Tensor[[64, 64], pl.FP32]
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            a: pl.Tile[[64, 64], pl.FP32] = pl.op.block.load(t, 0, 0, 64, 64)
            b: pl.Tile[[64, 64], pl.FP32] = pl.op.block.exp(a)
            result: pl.Tensor[[64, 64], pl.FP32] = pl.op.block.store(b, 0, 0, 64, 64, out)
            return result

        ir.assert_structural_equal(unified, explicit)

    def test_matmul(self):
        @pl.function
        def unified(
            t1: pl.Tensor[[64, 64], pl.FP16],
            t2: pl.Tensor[[64, 64], pl.FP16],
            out: pl.Tensor[[64, 64], pl.FP16],
        ) -> pl.Tensor[[64, 64], pl.FP16]:
            a: pl.Tile[[64, 64], pl.FP16] = pl.op.block.load(t1, 0, 0, 64, 64)
            b: pl.Tile[[64, 64], pl.FP16] = pl.op.block.load(t2, 0, 0, 64, 64)
            c: pl.Tile[[64, 64], pl.FP16] = pl.op.matmul(a, b)
            result: pl.Tensor[[64, 64], pl.FP16] = pl.op.block.store(c, 0, 0, 64, 64, out)
            return result

        @pl.function
        def explicit(
            t1: pl.Tensor[[64, 64], pl.FP16],
            t2: pl.Tensor[[64, 64], pl.FP16],
            out: pl.Tensor[[64, 64], pl.FP16],
        ) -> pl.Tensor[[64, 64], pl.FP16]:
            a: pl.Tile[[64, 64], pl.FP16] = pl.op.block.load(t1, 0, 0, 64, 64)
            b: pl.Tile[[64, 64], pl.FP16] = pl.op.block.load(t2, 0, 0, 64, 64)
            c: pl.Tile[[64, 64], pl.FP16] = pl.op.block.matmul(a, b)
            result: pl.Tensor[[64, 64], pl.FP16] = pl.op.block.store(c, 0, 0, 64, 64, out)
            return result

        ir.assert_structural_equal(unified, explicit)

    def test_row_sum(self):
        @pl.function
        def unified(
            t: pl.Tensor[[64, 64], pl.FP32], out: pl.Tensor[[64, 64], pl.FP32]
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            a: pl.Tile[[64, 64], pl.FP32] = pl.op.block.load(t, 0, 0, 64, 64)
            b: pl.Tile[[64, 1], pl.FP32] = pl.op.row_sum(a)
            result: pl.Tensor[[64, 64], pl.FP32] = pl.op.block.store(b, 0, 0, 64, 1, out)
            return result

        @pl.function
        def explicit(
            t: pl.Tensor[[64, 64], pl.FP32], out: pl.Tensor[[64, 64], pl.FP32]
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            a: pl.Tile[[64, 64], pl.FP32] = pl.op.block.load(t, 0, 0, 64, 64)
            b: pl.Tile[[64, 1], pl.FP32] = pl.op.block.row_sum(a)
            result: pl.Tensor[[64, 64], pl.FP32] = pl.op.block.store(b, 0, 0, 64, 1, out)
            return result

        ir.assert_structural_equal(unified, explicit)


class TestScalarAutoDispatch:
    """pl.op.add(Tile, scalar) produces the same IR as pl.op.block.adds."""

    def test_add_tile_scalar(self):
        @pl.function
        def unified(
            t: pl.Tensor[[64, 64], pl.FP32], out: pl.Tensor[[64, 64], pl.FP32]
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            a: pl.Tile[[64, 64], pl.FP32] = pl.op.block.load(t, 0, 0, 64, 64)
            b: pl.Tile[[64, 64], pl.FP32] = pl.op.add(a, 5)
            result: pl.Tensor[[64, 64], pl.FP32] = pl.op.block.store(b, 0, 0, 64, 64, out)
            return result

        @pl.function
        def explicit(
            t: pl.Tensor[[64, 64], pl.FP32], out: pl.Tensor[[64, 64], pl.FP32]
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            a: pl.Tile[[64, 64], pl.FP32] = pl.op.block.load(t, 0, 0, 64, 64)
            b: pl.Tile[[64, 64], pl.FP32] = pl.op.block.adds(a, 5)
            result: pl.Tensor[[64, 64], pl.FP32] = pl.op.block.store(b, 0, 0, 64, 64, out)
            return result

        ir.assert_structural_equal(unified, explicit)

    def test_mul_tile_scalar(self):
        @pl.function
        def unified(
            t: pl.Tensor[[64, 64], pl.FP32], out: pl.Tensor[[64, 64], pl.FP32]
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            a: pl.Tile[[64, 64], pl.FP32] = pl.op.block.load(t, 0, 0, 64, 64)
            b: pl.Tile[[64, 64], pl.FP32] = pl.op.mul(a, 3.14)
            result: pl.Tensor[[64, 64], pl.FP32] = pl.op.block.store(b, 0, 0, 64, 64, out)
            return result

        @pl.function
        def explicit(
            t: pl.Tensor[[64, 64], pl.FP32], out: pl.Tensor[[64, 64], pl.FP32]
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            a: pl.Tile[[64, 64], pl.FP32] = pl.op.block.load(t, 0, 0, 64, 64)
            b: pl.Tile[[64, 64], pl.FP32] = pl.op.block.muls(a, 3.14)
            result: pl.Tensor[[64, 64], pl.FP32] = pl.op.block.store(b, 0, 0, 64, 64, out)
            return result

        ir.assert_structural_equal(unified, explicit)

    def test_sub_tile_scalar(self):
        @pl.function
        def unified(
            t: pl.Tensor[[64, 64], pl.FP32], out: pl.Tensor[[64, 64], pl.FP32]
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            a: pl.Tile[[64, 64], pl.FP32] = pl.op.block.load(t, 0, 0, 64, 64)
            b: pl.Tile[[64, 64], pl.FP32] = pl.op.sub(a, 2)
            result: pl.Tensor[[64, 64], pl.FP32] = pl.op.block.store(b, 0, 0, 64, 64, out)
            return result

        @pl.function
        def explicit(
            t: pl.Tensor[[64, 64], pl.FP32], out: pl.Tensor[[64, 64], pl.FP32]
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            a: pl.Tile[[64, 64], pl.FP32] = pl.op.block.load(t, 0, 0, 64, 64)
            b: pl.Tile[[64, 64], pl.FP32] = pl.op.block.subs(a, 2)
            result: pl.Tensor[[64, 64], pl.FP32] = pl.op.block.store(b, 0, 0, 64, 64, out)
            return result

        ir.assert_structural_equal(unified, explicit)

    def test_div_tile_scalar(self):
        @pl.function
        def unified(
            t: pl.Tensor[[64, 64], pl.FP32], out: pl.Tensor[[64, 64], pl.FP32]
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            a: pl.Tile[[64, 64], pl.FP32] = pl.op.block.load(t, 0, 0, 64, 64)
            b: pl.Tile[[64, 64], pl.FP32] = pl.op.div(a, 4)
            result: pl.Tensor[[64, 64], pl.FP32] = pl.op.block.store(b, 0, 0, 64, 64, out)
            return result

        @pl.function
        def explicit(
            t: pl.Tensor[[64, 64], pl.FP32], out: pl.Tensor[[64, 64], pl.FP32]
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            a: pl.Tile[[64, 64], pl.FP32] = pl.op.block.load(t, 0, 0, 64, 64)
            b: pl.Tile[[64, 64], pl.FP32] = pl.op.block.divs(a, 4)
            result: pl.Tensor[[64, 64], pl.FP32] = pl.op.block.store(b, 0, 0, 64, 64, out)
            return result

        ir.assert_structural_equal(unified, explicit)


class TestPromotedOps:
    """Promoted single-module ops produce the same IR as their explicit form."""

    def test_promoted_create(self):
        @pl.function
        def unified(a: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            c: pl.Tensor[[64], pl.FP32] = pl.op.create([64], dtype=pl.FP32)
            return c

        @pl.function
        def explicit(a: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            c: pl.Tensor[[64], pl.FP32] = pl.op.tensor.create([64], dtype=pl.FP32)
            return c

        ir.assert_structural_equal(unified, explicit)

    def test_promoted_load_store(self):
        @pl.function
        def unified(
            t: pl.Tensor[[64, 64], pl.FP32], out: pl.Tensor[[64, 64], pl.FP32]
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            a: pl.Tile[[64, 64], pl.FP32] = pl.op.load(t, 0, 0, 64, 64)
            result: pl.Tensor[[64, 64], pl.FP32] = pl.op.store(a, 0, 0, 64, 64, out)
            return result

        @pl.function
        def explicit(
            t: pl.Tensor[[64, 64], pl.FP32], out: pl.Tensor[[64, 64], pl.FP32]
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            a: pl.Tile[[64, 64], pl.FP32] = pl.op.block.load(t, 0, 0, 64, 64)
            result: pl.Tensor[[64, 64], pl.FP32] = pl.op.block.store(a, 0, 0, 64, 64, out)
            return result

        ir.assert_structural_equal(unified, explicit)


class TestUnifiedOpsTypeErrors:
    """Passing invalid types to unified_ops raises TypeError."""

    def test_add_invalid_lhs(self):
        with pytest.raises(TypeError, match="expected Tensor or Tile"):
            unified_ops.add("not_a_tensor", 1)  # type: ignore

    def test_mul_invalid_lhs(self):
        with pytest.raises(TypeError, match="expected Tensor or Tile"):
            unified_ops.mul(42, 2)  # type: ignore

    def test_exp_invalid_input(self):
        with pytest.raises(TypeError, match="expected Tensor or Tile"):
            unified_ops.exp("bad")  # type: ignore

    def test_reshape_invalid_input(self):
        with pytest.raises(TypeError, match="expected Tensor or Tile"):
            unified_ops.reshape(123, [4, 4])  # type: ignore

    def test_matmul_invalid_lhs(self):
        with pytest.raises(TypeError, match="expected Tensor or Tile"):
            unified_ops.matmul(1, 2)  # type: ignore


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
