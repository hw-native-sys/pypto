# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for unified operation dispatch (pl.*).

Each test builds two functions — one using the unified ``pl.X`` API and one
using the explicit ``pl.tensor.X`` / ``pl.block.X`` API — then asserts
they produce structurally equal IR.
"""

import pypto.language as pl
import pytest
from pypto import ir
from pypto.language.op import unified_ops


class TestUnifiedTensorDispatch:
    """pl.X with Tensor args produces the same IR as pl.tensor.X."""

    def test_add(self):
        @pl.function
        def unified(a: pl.Tensor[[64], pl.FP32], b: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            c: pl.Tensor[[64], pl.FP32] = pl.add(a, b)
            return c

        @pl.function
        def explicit(a: pl.Tensor[[64], pl.FP32], b: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            c: pl.Tensor[[64], pl.FP32] = pl.tensor.add(a, b)
            return c

        ir.assert_structural_equal(unified, explicit)

    def test_sub(self):
        @pl.function
        def unified(a: pl.Tensor[[64], pl.FP32], b: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            c: pl.Tensor[[64], pl.FP32] = pl.sub(a, b)
            return c

        @pl.function
        def explicit(a: pl.Tensor[[64], pl.FP32], b: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            c: pl.Tensor[[64], pl.FP32] = pl.tensor.sub(a, b)
            return c

        ir.assert_structural_equal(unified, explicit)

    def test_mul(self):
        @pl.function
        def unified(a: pl.Tensor[[64], pl.FP32], b: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            c: pl.Tensor[[64], pl.FP32] = pl.mul(a, b)
            return c

        @pl.function
        def explicit(a: pl.Tensor[[64], pl.FP32], b: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            c: pl.Tensor[[64], pl.FP32] = pl.tensor.mul(a, b)
            return c

        ir.assert_structural_equal(unified, explicit)

    def test_div(self):
        @pl.function
        def unified(a: pl.Tensor[[64], pl.FP32], b: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            c: pl.Tensor[[64], pl.FP32] = pl.div(a, b)
            return c

        @pl.function
        def explicit(a: pl.Tensor[[64], pl.FP32], b: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            c: pl.Tensor[[64], pl.FP32] = pl.tensor.div(a, b)
            return c

        ir.assert_structural_equal(unified, explicit)

    def test_maximum(self):
        @pl.function
        def unified(a: pl.Tensor[[64], pl.FP32], b: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            c: pl.Tensor[[64], pl.FP32] = pl.maximum(a, b)
            return c

        @pl.function
        def explicit(a: pl.Tensor[[64], pl.FP32], b: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            c: pl.Tensor[[64], pl.FP32] = pl.tensor.maximum(a, b)
            return c

        ir.assert_structural_equal(unified, explicit)

    def test_exp(self):
        @pl.function
        def unified(a: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            c: pl.Tensor[[64], pl.FP32] = pl.exp(a)
            return c

        @pl.function
        def explicit(a: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            c: pl.Tensor[[64], pl.FP32] = pl.tensor.exp(a)
            return c

        ir.assert_structural_equal(unified, explicit)

    def test_add_scalar(self):
        @pl.function
        def unified(a: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            c: pl.Tensor[[64], pl.FP32] = pl.add(a, 5)
            return c

        @pl.function
        def explicit(a: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            c: pl.Tensor[[64], pl.FP32] = pl.tensor.add(a, 5)
            return c

        ir.assert_structural_equal(unified, explicit)

    def test_matmul(self):
        @pl.function
        def unified(
            a: pl.Tensor[[64, 128], pl.FP16], b: pl.Tensor[[128, 64], pl.FP16]
        ) -> pl.Tensor[[64, 64], pl.FP16]:
            c: pl.Tensor[[64, 64], pl.FP16] = pl.matmul(a, b)
            return c

        @pl.function
        def explicit(
            a: pl.Tensor[[64, 128], pl.FP16], b: pl.Tensor[[128, 64], pl.FP16]
        ) -> pl.Tensor[[64, 64], pl.FP16]:
            c: pl.Tensor[[64, 64], pl.FP16] = pl.tensor.matmul(a, b)
            return c

        ir.assert_structural_equal(unified, explicit)

    def test_row_max(self):
        @pl.function
        def unified(a: pl.Tensor[[64, 128], pl.FP32]) -> pl.Tensor[[64, 1], pl.FP32]:
            c: pl.Tensor[[64, 1], pl.FP32] = pl.row_max(a)
            return c

        @pl.function
        def explicit(a: pl.Tensor[[64, 128], pl.FP32]) -> pl.Tensor[[64, 1], pl.FP32]:
            c: pl.Tensor[[64, 1], pl.FP32] = pl.tensor.row_max(a)
            return c

        ir.assert_structural_equal(unified, explicit)

    def test_row_sum(self):
        @pl.function
        def unified(a: pl.Tensor[[64, 128], pl.FP32]) -> pl.Tensor[[64, 1], pl.FP32]:
            c: pl.Tensor[[64, 1], pl.FP32] = pl.row_sum(a)
            return c

        @pl.function
        def explicit(a: pl.Tensor[[64, 128], pl.FP32]) -> pl.Tensor[[64, 1], pl.FP32]:
            c: pl.Tensor[[64, 1], pl.FP32] = pl.tensor.row_sum(a)
            return c

        ir.assert_structural_equal(unified, explicit)

    def test_reshape(self):
        @pl.function
        def unified(a: pl.Tensor[[64, 128], pl.FP32]) -> pl.Tensor[[128, 64], pl.FP32]:
            c: pl.Tensor[[128, 64], pl.FP32] = pl.reshape(a, [128, 64])
            return c

        @pl.function
        def explicit(a: pl.Tensor[[64, 128], pl.FP32]) -> pl.Tensor[[128, 64], pl.FP32]:
            c: pl.Tensor[[128, 64], pl.FP32] = pl.tensor.reshape(a, [128, 64])
            return c

        ir.assert_structural_equal(unified, explicit)


class TestUnifiedBlockDispatch:
    """pl.X with Tile args produces the same IR as pl.block.X."""

    def test_add(self):
        @pl.function
        def unified(
            t: pl.Tensor[[64, 64], pl.FP32], out: pl.Tensor[[64, 64], pl.FP32]
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            a: pl.Tile[[64, 64], pl.FP32] = pl.block.load(t, offsets=[0, 0], shapes=[64, 64])
            b: pl.Tile[[64, 64], pl.FP32] = pl.block.load(t, offsets=[0, 0], shapes=[64, 64])
            c: pl.Tile[[64, 64], pl.FP32] = pl.add(a, b)
            result: pl.Tensor[[64, 64], pl.FP32] = pl.block.store(
                c, offsets=[0, 0], shapes=[64, 64], output_tensor=out
            )
            return result

        @pl.function
        def explicit(
            t: pl.Tensor[[64, 64], pl.FP32], out: pl.Tensor[[64, 64], pl.FP32]
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            a: pl.Tile[[64, 64], pl.FP32] = pl.block.load(t, offsets=[0, 0], shapes=[64, 64])
            b: pl.Tile[[64, 64], pl.FP32] = pl.block.load(t, offsets=[0, 0], shapes=[64, 64])
            c: pl.Tile[[64, 64], pl.FP32] = pl.block.add(a, b)
            result: pl.Tensor[[64, 64], pl.FP32] = pl.block.store(
                c, offsets=[0, 0], shapes=[64, 64], output_tensor=out
            )
            return result

        ir.assert_structural_equal(unified, explicit)

    def test_sub(self):
        @pl.function
        def unified(
            t: pl.Tensor[[64, 64], pl.FP32], out: pl.Tensor[[64, 64], pl.FP32]
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            a: pl.Tile[[64, 64], pl.FP32] = pl.block.load(t, offsets=[0, 0], shapes=[64, 64])
            b: pl.Tile[[64, 64], pl.FP32] = pl.block.load(t, offsets=[0, 0], shapes=[64, 64])
            c: pl.Tile[[64, 64], pl.FP32] = pl.sub(a, b)
            result: pl.Tensor[[64, 64], pl.FP32] = pl.block.store(
                c, offsets=[0, 0], shapes=[64, 64], output_tensor=out
            )
            return result

        @pl.function
        def explicit(
            t: pl.Tensor[[64, 64], pl.FP32], out: pl.Tensor[[64, 64], pl.FP32]
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            a: pl.Tile[[64, 64], pl.FP32] = pl.block.load(t, offsets=[0, 0], shapes=[64, 64])
            b: pl.Tile[[64, 64], pl.FP32] = pl.block.load(t, offsets=[0, 0], shapes=[64, 64])
            c: pl.Tile[[64, 64], pl.FP32] = pl.block.sub(a, b)
            result: pl.Tensor[[64, 64], pl.FP32] = pl.block.store(
                c, offsets=[0, 0], shapes=[64, 64], output_tensor=out
            )
            return result

        ir.assert_structural_equal(unified, explicit)

    def test_exp(self):
        @pl.function
        def unified(
            t: pl.Tensor[[64, 64], pl.FP32], out: pl.Tensor[[64, 64], pl.FP32]
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            a: pl.Tile[[64, 64], pl.FP32] = pl.block.load(t, offsets=[0, 0], shapes=[64, 64])
            b: pl.Tile[[64, 64], pl.FP32] = pl.exp(a)
            result: pl.Tensor[[64, 64], pl.FP32] = pl.block.store(
                b, offsets=[0, 0], shapes=[64, 64], output_tensor=out
            )
            return result

        @pl.function
        def explicit(
            t: pl.Tensor[[64, 64], pl.FP32], out: pl.Tensor[[64, 64], pl.FP32]
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            a: pl.Tile[[64, 64], pl.FP32] = pl.block.load(t, offsets=[0, 0], shapes=[64, 64])
            b: pl.Tile[[64, 64], pl.FP32] = pl.block.exp(a)
            result: pl.Tensor[[64, 64], pl.FP32] = pl.block.store(
                b, offsets=[0, 0], shapes=[64, 64], output_tensor=out
            )
            return result

        ir.assert_structural_equal(unified, explicit)

    def test_matmul(self):
        @pl.function
        def unified(
            t1: pl.Tensor[[64, 64], pl.FP16],
            t2: pl.Tensor[[64, 64], pl.FP16],
            out: pl.Tensor[[64, 64], pl.FP16],
        ) -> pl.Tensor[[64, 64], pl.FP16]:
            a: pl.Tile[[64, 64], pl.FP16] = pl.block.load(t1, offsets=[0, 0], shapes=[64, 64])
            b: pl.Tile[[64, 64], pl.FP16] = pl.block.load(t2, offsets=[0, 0], shapes=[64, 64])
            c: pl.Tile[[64, 64], pl.FP16] = pl.matmul(a, b)
            result: pl.Tensor[[64, 64], pl.FP16] = pl.block.store(
                c, offsets=[0, 0], shapes=[64, 64], output_tensor=out
            )
            return result

        @pl.function
        def explicit(
            t1: pl.Tensor[[64, 64], pl.FP16],
            t2: pl.Tensor[[64, 64], pl.FP16],
            out: pl.Tensor[[64, 64], pl.FP16],
        ) -> pl.Tensor[[64, 64], pl.FP16]:
            a: pl.Tile[[64, 64], pl.FP16] = pl.block.load(t1, offsets=[0, 0], shapes=[64, 64])
            b: pl.Tile[[64, 64], pl.FP16] = pl.block.load(t2, offsets=[0, 0], shapes=[64, 64])
            c: pl.Tile[[64, 64], pl.FP16] = pl.block.matmul(a, b)
            result: pl.Tensor[[64, 64], pl.FP16] = pl.block.store(
                c, offsets=[0, 0], shapes=[64, 64], output_tensor=out
            )
            return result

        ir.assert_structural_equal(unified, explicit)

    def test_row_sum(self):
        @pl.function
        def unified(
            t: pl.Tensor[[64, 64], pl.FP32], out: pl.Tensor[[64, 64], pl.FP32]
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            a: pl.Tile[[64, 64], pl.FP32] = pl.block.load(t, offsets=[0, 0], shapes=[64, 64])
            tmp: pl.Tile[[64, 16], pl.FP32] = pl.block.create_tile(
                [64, 16], dtype=pl.FP32, target_memory=pl.MemorySpace.UB
            )
            b: pl.Tile[[64, 1], pl.FP32] = pl.row_sum(a, tmp)
            result: pl.Tensor[[64, 64], pl.FP32] = pl.block.store(
                b, offsets=[0, 0], shapes=[64, 1], output_tensor=out
            )
            return result

        @pl.function
        def explicit(
            t: pl.Tensor[[64, 64], pl.FP32], out: pl.Tensor[[64, 64], pl.FP32]
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            a: pl.Tile[[64, 64], pl.FP32] = pl.block.load(t, offsets=[0, 0], shapes=[64, 64])
            tmp: pl.Tile[[64, 16], pl.FP32] = pl.block.create_tile(
                [64, 16], dtype=pl.FP32, target_memory=pl.MemorySpace.UB
            )
            b: pl.Tile[[64, 1], pl.FP32] = pl.block.row_sum(a, tmp)
            result: pl.Tensor[[64, 64], pl.FP32] = pl.block.store(
                b, offsets=[0, 0], shapes=[64, 1], output_tensor=out
            )
            return result

        ir.assert_structural_equal(unified, explicit)


class TestScalarAutoDispatch:
    """pl.add(Tile, scalar) produces the same IR as pl.block.adds."""

    def test_add_tile_scalar(self):
        @pl.function
        def unified(
            t: pl.Tensor[[64, 64], pl.FP32], out: pl.Tensor[[64, 64], pl.FP32]
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            a: pl.Tile[[64, 64], pl.FP32] = pl.block.load(t, offsets=[0, 0], shapes=[64, 64])
            b: pl.Tile[[64, 64], pl.FP32] = pl.add(a, 5)
            result: pl.Tensor[[64, 64], pl.FP32] = pl.block.store(
                b, offsets=[0, 0], shapes=[64, 64], output_tensor=out
            )
            return result

        @pl.function
        def explicit(
            t: pl.Tensor[[64, 64], pl.FP32], out: pl.Tensor[[64, 64], pl.FP32]
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            a: pl.Tile[[64, 64], pl.FP32] = pl.block.load(t, offsets=[0, 0], shapes=[64, 64])
            b: pl.Tile[[64, 64], pl.FP32] = pl.block.adds(a, 5)
            result: pl.Tensor[[64, 64], pl.FP32] = pl.block.store(
                b, offsets=[0, 0], shapes=[64, 64], output_tensor=out
            )
            return result

        ir.assert_structural_equal(unified, explicit)

    def test_mul_tile_scalar(self):
        @pl.function
        def unified(
            t: pl.Tensor[[64, 64], pl.FP32], out: pl.Tensor[[64, 64], pl.FP32]
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            a: pl.Tile[[64, 64], pl.FP32] = pl.block.load(t, offsets=[0, 0], shapes=[64, 64])
            b: pl.Tile[[64, 64], pl.FP32] = pl.mul(a, 3.14)
            result: pl.Tensor[[64, 64], pl.FP32] = pl.block.store(
                b, offsets=[0, 0], shapes=[64, 64], output_tensor=out
            )
            return result

        @pl.function
        def explicit(
            t: pl.Tensor[[64, 64], pl.FP32], out: pl.Tensor[[64, 64], pl.FP32]
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            a: pl.Tile[[64, 64], pl.FP32] = pl.block.load(t, offsets=[0, 0], shapes=[64, 64])
            b: pl.Tile[[64, 64], pl.FP32] = pl.block.muls(a, 3.14)
            result: pl.Tensor[[64, 64], pl.FP32] = pl.block.store(
                b, offsets=[0, 0], shapes=[64, 64], output_tensor=out
            )
            return result

        ir.assert_structural_equal(unified, explicit)

    def test_sub_tile_scalar(self):
        @pl.function
        def unified(
            t: pl.Tensor[[64, 64], pl.FP32], out: pl.Tensor[[64, 64], pl.FP32]
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            a: pl.Tile[[64, 64], pl.FP32] = pl.block.load(t, offsets=[0, 0], shapes=[64, 64])
            b: pl.Tile[[64, 64], pl.FP32] = pl.sub(a, 2)
            result: pl.Tensor[[64, 64], pl.FP32] = pl.block.store(
                b, offsets=[0, 0], shapes=[64, 64], output_tensor=out
            )
            return result

        @pl.function
        def explicit(
            t: pl.Tensor[[64, 64], pl.FP32], out: pl.Tensor[[64, 64], pl.FP32]
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            a: pl.Tile[[64, 64], pl.FP32] = pl.block.load(t, offsets=[0, 0], shapes=[64, 64])
            b: pl.Tile[[64, 64], pl.FP32] = pl.block.subs(a, 2)
            result: pl.Tensor[[64, 64], pl.FP32] = pl.block.store(
                b, offsets=[0, 0], shapes=[64, 64], output_tensor=out
            )
            return result

        ir.assert_structural_equal(unified, explicit)

    def test_div_tile_scalar(self):
        @pl.function
        def unified(
            t: pl.Tensor[[64, 64], pl.FP32], out: pl.Tensor[[64, 64], pl.FP32]
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            a: pl.Tile[[64, 64], pl.FP32] = pl.block.load(t, offsets=[0, 0], shapes=[64, 64])
            b: pl.Tile[[64, 64], pl.FP32] = pl.div(a, 4)
            result: pl.Tensor[[64, 64], pl.FP32] = pl.block.store(
                b, offsets=[0, 0], shapes=[64, 64], output_tensor=out
            )
            return result

        @pl.function
        def explicit(
            t: pl.Tensor[[64, 64], pl.FP32], out: pl.Tensor[[64, 64], pl.FP32]
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            a: pl.Tile[[64, 64], pl.FP32] = pl.block.load(t, offsets=[0, 0], shapes=[64, 64])
            b: pl.Tile[[64, 64], pl.FP32] = pl.block.divs(a, 4)
            result: pl.Tensor[[64, 64], pl.FP32] = pl.block.store(
                b, offsets=[0, 0], shapes=[64, 64], output_tensor=out
            )
            return result

        ir.assert_structural_equal(unified, explicit)


class TestPromotedOps:
    """Promoted single-module ops produce the same IR as their explicit form."""

    def test_promoted_create(self):
        @pl.function
        def unified(a: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            c: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
            return c

        @pl.function
        def explicit(a: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            c: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
            return c

        ir.assert_structural_equal(unified, explicit)

    def test_promoted_dim(self):
        @pl.function
        def unified(a: pl.Tensor[[64, 128], pl.FP32]) -> pl.Scalar[pl.INT64]:
            d: pl.Scalar[pl.INT64] = pl.dim(a, 0)
            return d

        @pl.function
        def explicit(a: pl.Tensor[[64, 128], pl.FP32]) -> pl.Scalar[pl.INT64]:
            d: pl.Scalar[pl.INT64] = pl.dim(a, 0)
            return d

        ir.assert_structural_equal(unified, explicit)

    def test_promoted_load_store(self):
        @pl.function
        def unified(
            t: pl.Tensor[[64, 64], pl.FP32], out: pl.Tensor[[64, 64], pl.FP32]
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            a: pl.Tile[[64, 64], pl.FP32] = pl.load(t, offsets=[0, 0], shapes=[64, 64])
            result: pl.Tensor[[64, 64], pl.FP32] = pl.store(
                a, offsets=[0, 0], shapes=[64, 64], output_tensor=out
            )
            return result

        @pl.function
        def explicit(
            t: pl.Tensor[[64, 64], pl.FP32], out: pl.Tensor[[64, 64], pl.FP32]
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            a: pl.Tile[[64, 64], pl.FP32] = pl.block.load(t, offsets=[0, 0], shapes=[64, 64])
            result: pl.Tensor[[64, 64], pl.FP32] = pl.block.store(
                a, offsets=[0, 0], shapes=[64, 64], output_tensor=out
            )
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
