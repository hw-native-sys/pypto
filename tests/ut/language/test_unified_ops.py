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
using the explicit ``pl.tensor.X`` / ``pl.tile.X`` API — then asserts
they produce structurally equal IR.
"""

import os
import pathlib
import warnings

import pypto.ir.utils as pypto_ir_utils
import pypto.language as pl
import pypto.language.op as language_op
import pytest
from pypto import DataType, ir
from pypto.language.op import tile_ops, unified_ops
from pypto.language.typing import Scalar, Tensor, Tile


class TestUnifiedTensorDispatch:
    """pl.X with Tensor args produces the same IR as pl.tensor.X."""

    def _assert_explicit_tensor_scalar_sugar(self, op_name: str, scalar_val: int | float) -> None:
        """Assert explicit tensor scalar ops canonicalize to scalar-only forms."""
        if op_name == "add":

            @pl.function
            def sugared(a: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                c: pl.Tensor[[64], pl.FP32] = pl.tensor.add(a, scalar_val)
                return c

            @pl.function
            def canonical(a: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                c: pl.Tensor[[64], pl.FP32] = pl.tensor.adds(a, scalar_val)
                return c
        elif op_name == "mul":

            @pl.function
            def sugared(a: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                c: pl.Tensor[[64], pl.FP32] = pl.tensor.mul(a, scalar_val)
                return c

            @pl.function
            def canonical(a: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                c: pl.Tensor[[64], pl.FP32] = pl.tensor.muls(a, scalar_val)
                return c
        elif op_name == "sub":

            @pl.function
            def sugared(a: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                c: pl.Tensor[[64], pl.FP32] = pl.tensor.sub(a, scalar_val)
                return c

            @pl.function
            def canonical(a: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                c: pl.Tensor[[64], pl.FP32] = pl.tensor.subs(a, scalar_val)
                return c
        elif op_name == "div":

            @pl.function
            def sugared(a: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                c: pl.Tensor[[64], pl.FP32] = pl.tensor.div(a, scalar_val)
                return c

            @pl.function
            def canonical(a: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                c: pl.Tensor[[64], pl.FP32] = pl.tensor.divs(a, scalar_val)
                return c
        else:
            raise AssertionError(f"Unsupported tensor scalar sugar op: {op_name}")

        ir.assert_structural_equal(sugared, canonical)

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

    @pytest.mark.parametrize("high_precision", [False, True])
    def test_div(self, high_precision):
        @pl.function
        def unified(a: pl.Tensor[[64], pl.FP32], b: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            c: pl.Tensor[[64], pl.FP32] = pl.div(a, b, high_precision=high_precision)
            return c

        @pl.function
        def explicit(a: pl.Tensor[[64], pl.FP32], b: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            c: pl.Tensor[[64], pl.FP32] = pl.tensor.div(a, b, high_precision=high_precision)
            return c

        ir.assert_structural_equal(unified, explicit)

    def test_log_high_precision_uses_unified_export(self):
        @pl.function
        def unified(a: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            c: pl.Tensor[[64], pl.FP32] = pl.log(a, high_precision=True)
            return c

        @pl.function
        def explicit(a: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            c: pl.Tensor[[64], pl.FP32] = pl.tensor.log(a, high_precision=True)
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

    def test_neg(self):
        @pl.function
        def unified(a: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            c: pl.Tensor[[64], pl.FP32] = pl.neg(a)
            return c

        @pl.function
        def explicit(a: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            c: pl.Tensor[[64], pl.FP32] = pl.tensor.neg(a)
            return c

        ir.assert_structural_equal(unified, explicit)

    def test_recip(self):
        @pl.function
        def unified(a: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            c: pl.Tensor[[64], pl.FP32] = pl.recip(a)
            return c

        @pl.function
        def explicit(a: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            c: pl.Tensor[[64], pl.FP32] = pl.tensor.recip(a)
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

    @pytest.mark.parametrize(
        ("op_name", "scalar_val"),
        [("add", 5), ("mul", 2.0), ("sub", 3), ("div", 4.0)],
    )
    def test_explicit_tensor_scalar_sugars_to_scalar_op(self, op_name: str, scalar_val: int | float):
        """Explicit tensor scalar ops sugar to scalar-only forms."""
        self._assert_explicit_tensor_scalar_sugar(op_name, scalar_val)

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

    def test_reinterpret_view(self):
        """Tensor dispatch preserves the input kind and optional shape."""
        span = ir.Span.unknown()
        data = Tensor(expr=ir.Var("data", ir.TensorType([8, 16], DataType.FP32), span))

        unified = pl.reinterpret_view(data, pl.INT16, shape=[4, 64])
        explicit = pl.tensor.reinterpret_view(data, pl.INT16, shape=[4, 64])

        assert isinstance(unified, Tensor)
        ir.assert_structural_equal(unified.unwrap(), explicit.unwrap())

    def test_row_min(self):
        @pl.function
        def unified(a: pl.Tensor[[64, 128], pl.FP32]) -> pl.Tensor[[64, 1], pl.FP32]:
            c: pl.Tensor[[64, 1], pl.FP32] = pl.row_min(a)
            return c

        @pl.function
        def explicit(a: pl.Tensor[[64, 128], pl.FP32]) -> pl.Tensor[[64, 1], pl.FP32]:
            c: pl.Tensor[[64, 1], pl.FP32] = pl.tensor.row_min(a)
            return c

        ir.assert_structural_equal(unified, explicit)

    def test_col_max(self):
        @pl.function
        def unified(a: pl.Tensor[[64, 128], pl.FP32]) -> pl.Tensor[[1, 128], pl.FP32]:
            c: pl.Tensor[[1, 128], pl.FP32] = pl.col_max(a)
            return c

        @pl.function
        def explicit(a: pl.Tensor[[64, 128], pl.FP32]) -> pl.Tensor[[1, 128], pl.FP32]:
            c: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.col_max(a)
            return c

        ir.assert_structural_equal(unified, explicit)

    def test_col_min(self):
        @pl.function
        def unified(a: pl.Tensor[[64, 128], pl.FP32]) -> pl.Tensor[[1, 128], pl.FP32]:
            c: pl.Tensor[[1, 128], pl.FP32] = pl.col_min(a)
            return c

        @pl.function
        def explicit(a: pl.Tensor[[64, 128], pl.FP32]) -> pl.Tensor[[1, 128], pl.FP32]:
            c: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.col_min(a)
            return c

        ir.assert_structural_equal(unified, explicit)

    def test_row_expand(self):
        @pl.function
        def unified(
            a: pl.Tensor[[64, 128], pl.FP32], rv: pl.Tensor[[64, 1], pl.FP32]
        ) -> pl.Tensor[[64, 128], pl.FP32]:
            c: pl.Tensor[[64, 128], pl.FP32] = pl.row_expand(a, rv)
            return c

        @pl.function
        def explicit(
            a: pl.Tensor[[64, 128], pl.FP32], rv: pl.Tensor[[64, 1], pl.FP32]
        ) -> pl.Tensor[[64, 128], pl.FP32]:
            c: pl.Tensor[[64, 128], pl.FP32] = pl.tensor.row_expand(a, rv)
            return c

        ir.assert_structural_equal(unified, explicit)

    def test_row_expand_add(self):
        @pl.function
        def unified(
            a: pl.Tensor[[64, 128], pl.FP32], rv: pl.Tensor[[64, 1], pl.FP32]
        ) -> pl.Tensor[[64, 128], pl.FP32]:
            c: pl.Tensor[[64, 128], pl.FP32] = pl.row_expand_add(a, rv)
            return c

        @pl.function
        def explicit(
            a: pl.Tensor[[64, 128], pl.FP32], rv: pl.Tensor[[64, 1], pl.FP32]
        ) -> pl.Tensor[[64, 128], pl.FP32]:
            c: pl.Tensor[[64, 128], pl.FP32] = pl.tensor.row_expand_add(a, rv)
            return c

        ir.assert_structural_equal(unified, explicit)

    def test_row_expand_sub(self):
        @pl.function
        def unified(
            a: pl.Tensor[[64, 128], pl.FP32], rv: pl.Tensor[[64, 1], pl.FP32]
        ) -> pl.Tensor[[64, 128], pl.FP32]:
            c: pl.Tensor[[64, 128], pl.FP32] = pl.row_expand_sub(a, rv)
            return c

        @pl.function
        def explicit(
            a: pl.Tensor[[64, 128], pl.FP32], rv: pl.Tensor[[64, 1], pl.FP32]
        ) -> pl.Tensor[[64, 128], pl.FP32]:
            c: pl.Tensor[[64, 128], pl.FP32] = pl.tensor.row_expand_sub(a, rv)
            return c

        ir.assert_structural_equal(unified, explicit)

    def test_col_expand(self):
        @pl.function
        def unified(
            a: pl.Tensor[[64, 128], pl.FP32], cv: pl.Tensor[[1, 128], pl.FP32]
        ) -> pl.Tensor[[64, 128], pl.FP32]:
            c: pl.Tensor[[64, 128], pl.FP32] = pl.col_expand(a, cv)
            return c

        @pl.function
        def explicit(
            a: pl.Tensor[[64, 128], pl.FP32], cv: pl.Tensor[[1, 128], pl.FP32]
        ) -> pl.Tensor[[64, 128], pl.FP32]:
            c: pl.Tensor[[64, 128], pl.FP32] = pl.tensor.col_expand(a, cv)
            return c

        ir.assert_structural_equal(unified, explicit)

    def test_col_expand_div(self):
        @pl.function
        def unified(
            a: pl.Tensor[[64, 128], pl.FP32], cv: pl.Tensor[[1, 128], pl.FP32]
        ) -> pl.Tensor[[64, 128], pl.FP32]:
            c: pl.Tensor[[64, 128], pl.FP32] = pl.col_expand_div(a, cv)
            return c

        @pl.function
        def explicit(
            a: pl.Tensor[[64, 128], pl.FP32], cv: pl.Tensor[[1, 128], pl.FP32]
        ) -> pl.Tensor[[64, 128], pl.FP32]:
            c: pl.Tensor[[64, 128], pl.FP32] = pl.tensor.col_expand_div(a, cv)
            return c

        ir.assert_structural_equal(unified, explicit)

    def test_col_expand_sub(self):
        @pl.function
        def unified(
            a: pl.Tensor[[64, 128], pl.FP32], cv: pl.Tensor[[1, 128], pl.FP32]
        ) -> pl.Tensor[[64, 128], pl.FP32]:
            c: pl.Tensor[[64, 128], pl.FP32] = pl.col_expand_sub(a, cv)
            return c

        @pl.function
        def explicit(
            a: pl.Tensor[[64, 128], pl.FP32], cv: pl.Tensor[[1, 128], pl.FP32]
        ) -> pl.Tensor[[64, 128], pl.FP32]:
            c: pl.Tensor[[64, 128], pl.FP32] = pl.tensor.col_expand_sub(a, cv)
            return c

        ir.assert_structural_equal(unified, explicit)

    def test_col_expand_add(self):
        @pl.function
        def unified(
            a: pl.Tensor[[64, 128], pl.FP32], cv: pl.Tensor[[1, 128], pl.FP32]
        ) -> pl.Tensor[[64, 128], pl.FP32]:
            c: pl.Tensor[[64, 128], pl.FP32] = pl.col_expand_add(a, cv)
            return c

        @pl.function
        def explicit(
            a: pl.Tensor[[64, 128], pl.FP32], cv: pl.Tensor[[1, 128], pl.FP32]
        ) -> pl.Tensor[[64, 128], pl.FP32]:
            c: pl.Tensor[[64, 128], pl.FP32] = pl.tensor.col_expand_add(a, cv)
            return c

        ir.assert_structural_equal(unified, explicit)

    def test_expands(self):
        @pl.function
        def unified(a: pl.Tensor[[64, 128], pl.FP32]) -> pl.Tensor[[64, 128], pl.FP32]:
            c: pl.Tensor[[64, 128], pl.FP32] = pl.expands(a, 1.0)
            return c

        @pl.function
        def explicit(a: pl.Tensor[[64, 128], pl.FP32]) -> pl.Tensor[[64, 128], pl.FP32]:
            c: pl.Tensor[[64, 128], pl.FP32] = pl.tensor.expands(a, 1.0)
            return c

        ir.assert_structural_equal(unified, explicit)


class TestUnifiedBlockDispatch:
    """pl.X with Tile args produces the same IR as pl.tile.X."""

    def test_reinterpret_view(self):
        """Tile dispatch preserves the input kind and auto-detected shape."""
        span = ir.Span.unknown()
        data = Tile(expr=ir.Var("data", ir.TileType([8, 16], DataType.FP32), span))

        unified = pl.reinterpret_view(data, pl.INT16)
        explicit = pl.tile.reinterpret_view(data, pl.INT16)

        assert isinstance(unified, Tile)
        ir.assert_structural_equal(unified.unwrap(), explicit.unwrap())

    def test_add(self):
        @pl.function
        def unified(
            t: pl.Tensor[[64, 64], pl.FP32], out: pl.Tensor[[64, 64], pl.FP32]
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            a: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(t, offsets=[0, 0], shapes=[64, 64])
            b: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(t, offsets=[0, 0], shapes=[64, 64])
            c: pl.Tile[[64, 64], pl.FP32] = pl.add(a, b)
            result: pl.Tensor[[64, 64], pl.FP32] = pl.tile.store(c, offsets=[0, 0], output_tensor=out)
            return result

        @pl.function
        def explicit(
            t: pl.Tensor[[64, 64], pl.FP32], out: pl.Tensor[[64, 64], pl.FP32]
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            a: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(t, offsets=[0, 0], shapes=[64, 64])
            b: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(t, offsets=[0, 0], shapes=[64, 64])
            c: pl.Tile[[64, 64], pl.FP32] = pl.tile.add(a, b)
            result: pl.Tensor[[64, 64], pl.FP32] = pl.tile.store(c, offsets=[0, 0], output_tensor=out)
            return result

        ir.assert_structural_equal(unified, explicit)

    def test_sub(self):
        @pl.function
        def unified(
            t: pl.Tensor[[64, 64], pl.FP32], out: pl.Tensor[[64, 64], pl.FP32]
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            a: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(t, offsets=[0, 0], shapes=[64, 64])
            b: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(t, offsets=[0, 0], shapes=[64, 64])
            c: pl.Tile[[64, 64], pl.FP32] = pl.sub(a, b)
            result: pl.Tensor[[64, 64], pl.FP32] = pl.tile.store(c, offsets=[0, 0], output_tensor=out)
            return result

        @pl.function
        def explicit(
            t: pl.Tensor[[64, 64], pl.FP32], out: pl.Tensor[[64, 64], pl.FP32]
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            a: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(t, offsets=[0, 0], shapes=[64, 64])
            b: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(t, offsets=[0, 0], shapes=[64, 64])
            c: pl.Tile[[64, 64], pl.FP32] = pl.tile.sub(a, b)
            result: pl.Tensor[[64, 64], pl.FP32] = pl.tile.store(c, offsets=[0, 0], output_tensor=out)
            return result

        ir.assert_structural_equal(unified, explicit)

    def test_exp(self):
        @pl.function
        def unified(
            t: pl.Tensor[[64, 64], pl.FP32], out: pl.Tensor[[64, 64], pl.FP32]
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            a: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(t, offsets=[0, 0], shapes=[64, 64])
            b: pl.Tile[[64, 64], pl.FP32] = pl.exp(a)
            result: pl.Tensor[[64, 64], pl.FP32] = pl.tile.store(b, offsets=[0, 0], output_tensor=out)
            return result

        @pl.function
        def explicit(
            t: pl.Tensor[[64, 64], pl.FP32], out: pl.Tensor[[64, 64], pl.FP32]
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            a: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(t, offsets=[0, 0], shapes=[64, 64])
            b: pl.Tile[[64, 64], pl.FP32] = pl.tile.exp(a)
            result: pl.Tensor[[64, 64], pl.FP32] = pl.tile.store(b, offsets=[0, 0], output_tensor=out)
            return result

        ir.assert_structural_equal(unified, explicit)

    def test_neg(self):
        @pl.function
        def unified(
            t: pl.Tensor[[64, 64], pl.FP32], out: pl.Tensor[[64, 64], pl.FP32]
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            a: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(t, offsets=[0, 0], shapes=[64, 64])
            b: pl.Tile[[64, 64], pl.FP32] = pl.neg(a)
            result: pl.Tensor[[64, 64], pl.FP32] = pl.tile.store(b, offsets=[0, 0], output_tensor=out)
            return result

        @pl.function
        def explicit(
            t: pl.Tensor[[64, 64], pl.FP32], out: pl.Tensor[[64, 64], pl.FP32]
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            a: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(t, offsets=[0, 0], shapes=[64, 64])
            b: pl.Tile[[64, 64], pl.FP32] = pl.tile.neg(a)
            result: pl.Tensor[[64, 64], pl.FP32] = pl.tile.store(b, offsets=[0, 0], output_tensor=out)
            return result

        ir.assert_structural_equal(unified, explicit)

    def test_recip(self):
        @pl.function
        def unified(
            t: pl.Tensor[[64, 64], pl.FP32], out: pl.Tensor[[64, 64], pl.FP32]
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            a: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(t, offsets=[0, 0], shapes=[64, 64])
            b: pl.Tile[[64, 64], pl.FP32] = pl.recip(a)
            result: pl.Tensor[[64, 64], pl.FP32] = pl.tile.store(b, offsets=[0, 0], output_tensor=out)
            return result

        @pl.function
        def explicit(
            t: pl.Tensor[[64, 64], pl.FP32], out: pl.Tensor[[64, 64], pl.FP32]
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            a: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(t, offsets=[0, 0], shapes=[64, 64])
            b: pl.Tile[[64, 64], pl.FP32] = pl.tile.recip(a)
            result: pl.Tensor[[64, 64], pl.FP32] = pl.tile.store(b, offsets=[0, 0], output_tensor=out)
            return result

        ir.assert_structural_equal(unified, explicit)

    def test_matmul(self):
        @pl.function
        def unified(
            t1: pl.Tensor[[64, 64], pl.FP16],
            t2: pl.Tensor[[64, 64], pl.FP16],
            out: pl.Tensor[[64, 64], pl.FP16],
        ) -> pl.Tensor[[64, 64], pl.FP16]:
            a: pl.Tile[[64, 64], pl.FP16] = pl.tile.load(t1, offsets=[0, 0], shapes=[64, 64])
            b: pl.Tile[[64, 64], pl.FP16] = pl.tile.load(t2, offsets=[0, 0], shapes=[64, 64])
            c: pl.Tile[[64, 64], pl.FP32] = pl.matmul(a, b)
            result: pl.Tensor[[64, 64], pl.FP16] = pl.tile.store(c, offsets=[0, 0], output_tensor=out)
            return result

        @pl.function
        def explicit(
            t1: pl.Tensor[[64, 64], pl.FP16],
            t2: pl.Tensor[[64, 64], pl.FP16],
            out: pl.Tensor[[64, 64], pl.FP16],
        ) -> pl.Tensor[[64, 64], pl.FP16]:
            a: pl.Tile[[64, 64], pl.FP16] = pl.tile.load(t1, offsets=[0, 0], shapes=[64, 64])
            b: pl.Tile[[64, 64], pl.FP16] = pl.tile.load(t2, offsets=[0, 0], shapes=[64, 64])
            c: pl.Tile[[64, 64], pl.FP32] = pl.tile.matmul(a, b)
            result: pl.Tensor[[64, 64], pl.FP16] = pl.tile.store(c, offsets=[0, 0], output_tensor=out)
            return result

        ir.assert_structural_equal(unified, explicit)

    def test_batch_matmul(self):
        @pl.function
        def unified(
            t1: pl.Tensor[[2, 64, 64], pl.FP16],
            t2: pl.Tensor[[2, 64, 64], pl.FP16],
            out: pl.Tensor[[2, 64, 64], pl.FP16],
        ) -> pl.Tensor[[2, 64, 64], pl.FP16]:
            a: pl.Tile[[2, 64, 64], pl.FP16] = pl.tile.load(t1, offsets=[0, 0, 0], shapes=[2, 64, 64])
            b: pl.Tile[[2, 64, 64], pl.FP16] = pl.tile.load(t2, offsets=[0, 0, 0], shapes=[2, 64, 64])
            c: pl.Tile[[2, 64, 64], pl.FP32] = pl.batch_matmul(a, b)
            result: pl.Tensor[[2, 64, 64], pl.FP16] = pl.tile.store(c, offsets=[0, 0, 0], output_tensor=out)
            return result

        @pl.function
        def explicit(
            t1: pl.Tensor[[2, 64, 64], pl.FP16],
            t2: pl.Tensor[[2, 64, 64], pl.FP16],
            out: pl.Tensor[[2, 64, 64], pl.FP16],
        ) -> pl.Tensor[[2, 64, 64], pl.FP16]:
            a: pl.Tile[[2, 64, 64], pl.FP16] = pl.tile.load(t1, offsets=[0, 0, 0], shapes=[2, 64, 64])
            b: pl.Tile[[2, 64, 64], pl.FP16] = pl.tile.load(t2, offsets=[0, 0, 0], shapes=[2, 64, 64])
            c: pl.Tile[[2, 64, 64], pl.FP32] = pl.tile.batch_matmul(a, b)
            result: pl.Tensor[[2, 64, 64], pl.FP16] = pl.tile.store(c, offsets=[0, 0, 0], output_tensor=out)
            return result

        ir.assert_structural_equal(unified, explicit)

    def test_row_sum(self):
        @pl.function
        def unified(
            t: pl.Tensor[[64, 64], pl.FP32], out: pl.Tensor[[64, 64], pl.FP32]
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            a: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(t, offsets=[0, 0], shapes=[64, 64])
            tmp: pl.Tile[[64, 64], pl.FP32] = pl.tile.create(
                [64, 64], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
            )
            b: pl.Tile[[64, 1], pl.FP32] = pl.row_sum(a, tmp)
            result: pl.Tensor[[64, 64], pl.FP32] = pl.tile.store(b, offsets=[0, 0], output_tensor=out)
            return result

        @pl.function
        def explicit(
            t: pl.Tensor[[64, 64], pl.FP32], out: pl.Tensor[[64, 64], pl.FP32]
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            a: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(t, offsets=[0, 0], shapes=[64, 64])
            tmp: pl.Tile[[64, 64], pl.FP32] = pl.tile.create(
                [64, 64], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
            )
            b: pl.Tile[[64, 1], pl.FP32] = pl.tile.row_sum(a, tmp)
            result: pl.Tensor[[64, 64], pl.FP32] = pl.tile.store(b, offsets=[0, 0], output_tensor=out)
            return result

        ir.assert_structural_equal(unified, explicit)

    def test_row_min(self):
        @pl.function
        def unified(
            t: pl.Tensor[[64, 64], pl.FP32], out: pl.Tensor[[64, 64], pl.FP32]
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            a: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(t, offsets=[0, 0], shapes=[64, 64])
            tmp: pl.Tile[[64, 64], pl.FP32] = pl.tile.create(
                [64, 64], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
            )
            b: pl.Tile[[64, 1], pl.FP32] = pl.row_min(a, tmp)
            result: pl.Tensor[[64, 64], pl.FP32] = pl.tile.store(b, offsets=[0, 0], output_tensor=out)
            return result

        @pl.function
        def explicit(
            t: pl.Tensor[[64, 64], pl.FP32], out: pl.Tensor[[64, 64], pl.FP32]
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            a: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(t, offsets=[0, 0], shapes=[64, 64])
            tmp: pl.Tile[[64, 64], pl.FP32] = pl.tile.create(
                [64, 64], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
            )
            b: pl.Tile[[64, 1], pl.FP32] = pl.tile.row_min(a, tmp)
            result: pl.Tensor[[64, 64], pl.FP32] = pl.tile.store(b, offsets=[0, 0], output_tensor=out)
            return result

        ir.assert_structural_equal(unified, explicit)

    def test_row_expand(self):
        @pl.function
        def unified(
            t: pl.Tensor[[64, 64], pl.FP32],
            row_t: pl.Tensor[[64, 64], pl.FP32],
            out: pl.Tensor[[64, 64], pl.FP32],
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            a: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(t, offsets=[0, 0], shapes=[64, 64])
            rv: pl.Tile[[64, 1], pl.FP32] = pl.tile.load(row_t, offsets=[0, 0], shapes=[64, 1])
            b: pl.Tile[[64, 64], pl.FP32] = pl.row_expand(a, rv)
            result: pl.Tensor[[64, 64], pl.FP32] = pl.tile.store(b, offsets=[0, 0], output_tensor=out)
            return result

        @pl.function
        def explicit(
            t: pl.Tensor[[64, 64], pl.FP32],
            row_t: pl.Tensor[[64, 64], pl.FP32],
            out: pl.Tensor[[64, 64], pl.FP32],
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            a: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(t, offsets=[0, 0], shapes=[64, 64])
            rv: pl.Tile[[64, 1], pl.FP32] = pl.tile.load(row_t, offsets=[0, 0], shapes=[64, 1])
            b: pl.Tile[[64, 64], pl.FP32] = pl.tile.row_expand(a, rv)
            result: pl.Tensor[[64, 64], pl.FP32] = pl.tile.store(b, offsets=[0, 0], output_tensor=out)
            return result

        ir.assert_structural_equal(unified, explicit)

    def test_row_expand_add_with_tmp(self):
        @pl.function
        def unified(
            t: pl.Tensor[[64, 64], pl.FP32], out: pl.Tensor[[64, 64], pl.FP32]
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            a: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(t, offsets=[0, 0], shapes=[64, 64])
            rv: pl.Tile[[64, 1], pl.FP32] = pl.tile.load(t, offsets=[0, 0], shapes=[64, 1])
            tmp: pl.Tile[[64, 64], pl.FP32] = pl.tile.create(
                [64, 64], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
            )
            b: pl.Tile[[64, 64], pl.FP32] = pl.row_expand_add(a, rv, tmp)
            result: pl.Tensor[[64, 64], pl.FP32] = pl.tile.store(b, offsets=[0, 0], output_tensor=out)
            return result

        @pl.function
        def explicit(
            t: pl.Tensor[[64, 64], pl.FP32], out: pl.Tensor[[64, 64], pl.FP32]
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            a: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(t, offsets=[0, 0], shapes=[64, 64])
            rv: pl.Tile[[64, 1], pl.FP32] = pl.tile.load(t, offsets=[0, 0], shapes=[64, 1])
            tmp: pl.Tile[[64, 64], pl.FP32] = pl.tile.create(
                [64, 64], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
            )
            b: pl.Tile[[64, 64], pl.FP32] = pl.tile.row_expand_add(a, rv, tmp)
            result: pl.Tensor[[64, 64], pl.FP32] = pl.tile.store(b, offsets=[0, 0], output_tensor=out)
            return result

        ir.assert_structural_equal(unified, explicit)

    def test_row_expand_add_without_tmp(self):
        """The Tile overload keeps the original two-operand dispatch."""
        span = ir.Span.unknown()
        lhs = Tile(expr=ir.Var("lhs", ir.TileType([8, 8], DataType.FP32), span))
        rhs = Tile(expr=ir.Var("rhs", ir.TileType([8, 1], DataType.FP32), span))

        unified = pl.row_expand_add(lhs, rhs)
        explicit = pl.tile.row_expand_add(lhs, rhs)

        ir.assert_structural_equal(unified.unwrap(), explicit.unwrap())

    def test_row_expand_sub(self):
        @pl.function
        def unified(
            t: pl.Tensor[[64, 64], pl.FP32], out: pl.Tensor[[64, 64], pl.FP32]
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            a: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(t, offsets=[0, 0], shapes=[64, 64])
            rv: pl.Tile[[64, 1], pl.FP32] = pl.tile.load(t, offsets=[0, 0], shapes=[64, 1])
            b: pl.Tile[[64, 64], pl.FP32] = pl.row_expand_sub(a, rv)
            result: pl.Tensor[[64, 64], pl.FP32] = pl.tile.store(b, offsets=[0, 0], output_tensor=out)
            return result

        @pl.function
        def explicit(
            t: pl.Tensor[[64, 64], pl.FP32], out: pl.Tensor[[64, 64], pl.FP32]
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            a: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(t, offsets=[0, 0], shapes=[64, 64])
            rv: pl.Tile[[64, 1], pl.FP32] = pl.tile.load(t, offsets=[0, 0], shapes=[64, 1])
            b: pl.Tile[[64, 64], pl.FP32] = pl.tile.row_expand_sub(a, rv)
            result: pl.Tensor[[64, 64], pl.FP32] = pl.tile.store(b, offsets=[0, 0], output_tensor=out)
            return result

        ir.assert_structural_equal(unified, explicit)

    def test_col_expand(self):
        @pl.function
        def unified(
            t: pl.Tensor[[64, 64], pl.FP32], out: pl.Tensor[[64, 64], pl.FP32]
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            a: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(t, offsets=[0, 0], shapes=[64, 64])
            cv: pl.Tile[[1, 64], pl.FP32] = pl.tile.load(t, offsets=[0, 0], shapes=[1, 64])
            b: pl.Tile[[64, 64], pl.FP32] = pl.col_expand(a, cv)
            result: pl.Tensor[[64, 64], pl.FP32] = pl.tile.store(b, offsets=[0, 0], output_tensor=out)
            return result

        @pl.function
        def explicit(
            t: pl.Tensor[[64, 64], pl.FP32], out: pl.Tensor[[64, 64], pl.FP32]
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            a: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(t, offsets=[0, 0], shapes=[64, 64])
            cv: pl.Tile[[1, 64], pl.FP32] = pl.tile.load(t, offsets=[0, 0], shapes=[1, 64])
            b: pl.Tile[[64, 64], pl.FP32] = pl.tile.col_expand(a, cv)
            result: pl.Tensor[[64, 64], pl.FP32] = pl.tile.store(b, offsets=[0, 0], output_tensor=out)
            return result

        ir.assert_structural_equal(unified, explicit)

    def test_col_expand_div(self):
        @pl.function
        def unified(
            t: pl.Tensor[[64, 64], pl.FP32], out: pl.Tensor[[64, 64], pl.FP32]
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            a: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(t, offsets=[0, 0], shapes=[64, 64])
            cv: pl.Tile[[1, 64], pl.FP32] = pl.tile.load(t, offsets=[0, 0], shapes=[1, 64])
            b: pl.Tile[[64, 64], pl.FP32] = pl.col_expand_div(a, cv)
            result: pl.Tensor[[64, 64], pl.FP32] = pl.tile.store(b, offsets=[0, 0], output_tensor=out)
            return result

        @pl.function
        def explicit(
            t: pl.Tensor[[64, 64], pl.FP32], out: pl.Tensor[[64, 64], pl.FP32]
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            a: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(t, offsets=[0, 0], shapes=[64, 64])
            cv: pl.Tile[[1, 64], pl.FP32] = pl.tile.load(t, offsets=[0, 0], shapes=[1, 64])
            b: pl.Tile[[64, 64], pl.FP32] = pl.tile.col_expand_div(a, cv)
            result: pl.Tensor[[64, 64], pl.FP32] = pl.tile.store(b, offsets=[0, 0], output_tensor=out)
            return result

        ir.assert_structural_equal(unified, explicit)

    def test_col_expand_sub(self):
        @pl.function
        def unified(
            t: pl.Tensor[[64, 64], pl.FP32], out: pl.Tensor[[64, 64], pl.FP32]
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            a: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(t, offsets=[0, 0], shapes=[64, 64])
            cv: pl.Tile[[1, 64], pl.FP32] = pl.tile.load(t, offsets=[0, 0], shapes=[1, 64])
            b: pl.Tile[[64, 64], pl.FP32] = pl.col_expand_sub(a, cv)
            result: pl.Tensor[[64, 64], pl.FP32] = pl.tile.store(b, offsets=[0, 0], output_tensor=out)
            return result

        @pl.function
        def explicit(
            t: pl.Tensor[[64, 64], pl.FP32], out: pl.Tensor[[64, 64], pl.FP32]
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            a: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(t, offsets=[0, 0], shapes=[64, 64])
            cv: pl.Tile[[1, 64], pl.FP32] = pl.tile.load(t, offsets=[0, 0], shapes=[1, 64])
            b: pl.Tile[[64, 64], pl.FP32] = pl.tile.col_expand_sub(a, cv)
            result: pl.Tensor[[64, 64], pl.FP32] = pl.tile.store(b, offsets=[0, 0], output_tensor=out)
            return result

        ir.assert_structural_equal(unified, explicit)

    def test_col_expand_add(self):
        @pl.function
        def unified(
            t: pl.Tensor[[64, 64], pl.FP32], out: pl.Tensor[[64, 64], pl.FP32]
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            a: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(t, offsets=[0, 0], shapes=[64, 64])
            cv: pl.Tile[[1, 64], pl.FP32] = pl.tile.load(t, offsets=[0, 0], shapes=[1, 64])
            b: pl.Tile[[64, 64], pl.FP32] = pl.col_expand_add(a, cv)
            result: pl.Tensor[[64, 64], pl.FP32] = pl.tile.store(b, offsets=[0, 0], output_tensor=out)
            return result

        @pl.function
        def explicit(
            t: pl.Tensor[[64, 64], pl.FP32], out: pl.Tensor[[64, 64], pl.FP32]
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            a: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(t, offsets=[0, 0], shapes=[64, 64])
            cv: pl.Tile[[1, 64], pl.FP32] = pl.tile.load(t, offsets=[0, 0], shapes=[1, 64])
            b: pl.Tile[[64, 64], pl.FP32] = pl.tile.col_expand_add(a, cv)
            result: pl.Tensor[[64, 64], pl.FP32] = pl.tile.store(b, offsets=[0, 0], output_tensor=out)
            return result

        ir.assert_structural_equal(unified, explicit)

    def test_expands(self):
        @pl.function
        def unified(
            t: pl.Tensor[[64, 64], pl.FP32], out: pl.Tensor[[64, 64], pl.FP32]
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            a: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(t, offsets=[0, 0], shapes=[64, 64])
            b: pl.Tile[[64, 64], pl.FP32] = pl.expands(a, 1.0)
            result: pl.Tensor[[64, 64], pl.FP32] = pl.tile.store(b, offsets=[0, 0], output_tensor=out)
            return result

        @pl.function
        def explicit(
            t: pl.Tensor[[64, 64], pl.FP32], out: pl.Tensor[[64, 64], pl.FP32]
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            a: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(t, offsets=[0, 0], shapes=[64, 64])
            b: pl.Tile[[64, 64], pl.FP32] = pl.tile.expands(a, 1.0)
            result: pl.Tensor[[64, 64], pl.FP32] = pl.tile.store(b, offsets=[0, 0], output_tensor=out)
            return result

        ir.assert_structural_equal(unified, explicit)


class TestScalarAutoDispatch:
    """pl.add(Tile, scalar) produces the same IR as pl.tile.adds."""

    def _assert_explicit_tile_scalar_sugar(self, op_name: str, scalar_val: int | float) -> None:
        """Assert explicit tile scalar ops canonicalize to scalar-only forms."""
        if op_name == "add":

            @pl.function
            def sugared(
                t: pl.Tensor[[64, 64], pl.FP32], out: pl.Tensor[[64, 64], pl.FP32]
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                a: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(t, offsets=[0, 0], shapes=[64, 64])
                b: pl.Tile[[64, 64], pl.FP32] = pl.tile.add(a, scalar_val)
                result: pl.Tensor[[64, 64], pl.FP32] = pl.tile.store(b, offsets=[0, 0], output_tensor=out)
                return result

            @pl.function
            def canonical(
                t: pl.Tensor[[64, 64], pl.FP32], out: pl.Tensor[[64, 64], pl.FP32]
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                a: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(t, offsets=[0, 0], shapes=[64, 64])
                b: pl.Tile[[64, 64], pl.FP32] = pl.tile.adds(a, scalar_val)
                result: pl.Tensor[[64, 64], pl.FP32] = pl.tile.store(b, offsets=[0, 0], output_tensor=out)
                return result
        elif op_name == "mul":

            @pl.function
            def sugared(
                t: pl.Tensor[[64, 64], pl.FP32], out: pl.Tensor[[64, 64], pl.FP32]
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                a: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(t, offsets=[0, 0], shapes=[64, 64])
                b: pl.Tile[[64, 64], pl.FP32] = pl.tile.mul(a, scalar_val)
                result: pl.Tensor[[64, 64], pl.FP32] = pl.tile.store(b, offsets=[0, 0], output_tensor=out)
                return result

            @pl.function
            def canonical(
                t: pl.Tensor[[64, 64], pl.FP32], out: pl.Tensor[[64, 64], pl.FP32]
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                a: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(t, offsets=[0, 0], shapes=[64, 64])
                b: pl.Tile[[64, 64], pl.FP32] = pl.tile.muls(a, scalar_val)
                result: pl.Tensor[[64, 64], pl.FP32] = pl.tile.store(b, offsets=[0, 0], output_tensor=out)
                return result
        elif op_name == "sub":

            @pl.function
            def sugared(
                t: pl.Tensor[[64, 64], pl.FP32], out: pl.Tensor[[64, 64], pl.FP32]
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                a: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(t, offsets=[0, 0], shapes=[64, 64])
                b: pl.Tile[[64, 64], pl.FP32] = pl.tile.sub(a, scalar_val)
                result: pl.Tensor[[64, 64], pl.FP32] = pl.tile.store(b, offsets=[0, 0], output_tensor=out)
                return result

            @pl.function
            def canonical(
                t: pl.Tensor[[64, 64], pl.FP32], out: pl.Tensor[[64, 64], pl.FP32]
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                a: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(t, offsets=[0, 0], shapes=[64, 64])
                b: pl.Tile[[64, 64], pl.FP32] = pl.tile.subs(a, scalar_val)
                result: pl.Tensor[[64, 64], pl.FP32] = pl.tile.store(b, offsets=[0, 0], output_tensor=out)
                return result
        elif op_name == "div":

            @pl.function
            def sugared(
                t: pl.Tensor[[64, 64], pl.FP32], out: pl.Tensor[[64, 64], pl.FP32]
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                a: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(t, offsets=[0, 0], shapes=[64, 64])
                b: pl.Tile[[64, 64], pl.FP32] = pl.tile.div(a, scalar_val)
                result: pl.Tensor[[64, 64], pl.FP32] = pl.tile.store(b, offsets=[0, 0], output_tensor=out)
                return result

            @pl.function
            def canonical(
                t: pl.Tensor[[64, 64], pl.FP32], out: pl.Tensor[[64, 64], pl.FP32]
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                a: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(t, offsets=[0, 0], shapes=[64, 64])
                b: pl.Tile[[64, 64], pl.FP32] = pl.tile.divs(a, scalar_val)
                result: pl.Tensor[[64, 64], pl.FP32] = pl.tile.store(b, offsets=[0, 0], output_tensor=out)
                return result
        else:
            raise AssertionError(f"Unsupported tile scalar sugar op: {op_name}")

        ir.assert_structural_equal(sugared, canonical)

    def test_add_tile_scalar(self):
        @pl.function
        def unified(
            t: pl.Tensor[[64, 64], pl.FP32], out: pl.Tensor[[64, 64], pl.FP32]
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            a: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(t, offsets=[0, 0], shapes=[64, 64])
            b: pl.Tile[[64, 64], pl.FP32] = pl.add(a, 5)
            result: pl.Tensor[[64, 64], pl.FP32] = pl.tile.store(b, offsets=[0, 0], output_tensor=out)
            return result

        @pl.function
        def explicit(
            t: pl.Tensor[[64, 64], pl.FP32], out: pl.Tensor[[64, 64], pl.FP32]
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            a: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(t, offsets=[0, 0], shapes=[64, 64])
            b: pl.Tile[[64, 64], pl.FP32] = pl.tile.adds(a, 5)
            result: pl.Tensor[[64, 64], pl.FP32] = pl.tile.store(b, offsets=[0, 0], output_tensor=out)
            return result

        ir.assert_structural_equal(unified, explicit)

    def test_mul_tile_scalar(self):
        @pl.function
        def unified(
            t: pl.Tensor[[64, 64], pl.FP32], out: pl.Tensor[[64, 64], pl.FP32]
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            a: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(t, offsets=[0, 0], shapes=[64, 64])
            b: pl.Tile[[64, 64], pl.FP32] = pl.mul(a, 3.14)
            result: pl.Tensor[[64, 64], pl.FP32] = pl.tile.store(b, offsets=[0, 0], output_tensor=out)
            return result

        @pl.function
        def explicit(
            t: pl.Tensor[[64, 64], pl.FP32], out: pl.Tensor[[64, 64], pl.FP32]
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            a: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(t, offsets=[0, 0], shapes=[64, 64])
            b: pl.Tile[[64, 64], pl.FP32] = pl.tile.muls(a, 3.14)
            result: pl.Tensor[[64, 64], pl.FP32] = pl.tile.store(b, offsets=[0, 0], output_tensor=out)
            return result

        ir.assert_structural_equal(unified, explicit)

    def test_sub_tile_scalar(self):
        @pl.function
        def unified(
            t: pl.Tensor[[64, 64], pl.FP32], out: pl.Tensor[[64, 64], pl.FP32]
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            a: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(t, offsets=[0, 0], shapes=[64, 64])
            b: pl.Tile[[64, 64], pl.FP32] = pl.sub(a, 2)
            result: pl.Tensor[[64, 64], pl.FP32] = pl.tile.store(b, offsets=[0, 0], output_tensor=out)
            return result

        @pl.function
        def explicit(
            t: pl.Tensor[[64, 64], pl.FP32], out: pl.Tensor[[64, 64], pl.FP32]
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            a: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(t, offsets=[0, 0], shapes=[64, 64])
            b: pl.Tile[[64, 64], pl.FP32] = pl.tile.subs(a, 2)
            result: pl.Tensor[[64, 64], pl.FP32] = pl.tile.store(b, offsets=[0, 0], output_tensor=out)
            return result

        ir.assert_structural_equal(unified, explicit)

    def test_div_tile_scalar(self):
        @pl.function
        def unified(
            t: pl.Tensor[[64, 64], pl.FP32], out: pl.Tensor[[64, 64], pl.FP32]
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            a: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(t, offsets=[0, 0], shapes=[64, 64])
            b: pl.Tile[[64, 64], pl.FP32] = pl.div(a, 4)
            result: pl.Tensor[[64, 64], pl.FP32] = pl.tile.store(b, offsets=[0, 0], output_tensor=out)
            return result

        @pl.function
        def explicit(
            t: pl.Tensor[[64, 64], pl.FP32], out: pl.Tensor[[64, 64], pl.FP32]
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            a: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(t, offsets=[0, 0], shapes=[64, 64])
            b: pl.Tile[[64, 64], pl.FP32] = pl.tile.divs(a, 4)
            result: pl.Tensor[[64, 64], pl.FP32] = pl.tile.store(b, offsets=[0, 0], output_tensor=out)
            return result

        ir.assert_structural_equal(unified, explicit)

    @pytest.mark.parametrize(
        ("op_name", "scalar_val"),
        [("add", 5), ("mul", 3.14), ("sub", 2), ("div", 4)],
    )
    def test_explicit_tile_scalar_sugars_to_scalar_op(self, op_name: str, scalar_val: int | float):
        """Explicit tile scalar ops sugar to scalar-only forms."""
        self._assert_explicit_tile_scalar_sugar(op_name, scalar_val)


class TestPromotedOps:
    """Promoted single-module ops produce the same IR as their explicit form."""

    def test_reinterpret_view_exports(self):
        assert pl.reinterpret_view is unified_ops.reinterpret_view
        assert language_op.reinterpret_view is unified_ops.reinterpret_view
        assert "reinterpret_view" in pl.__all__
        assert "reinterpret_view" in language_op.__all__

    def test_namespaces_agree_on_shared_names(self):
        """A DSL name must resolve to one object in ``pl`` and ``pl.op``.

        The parser resolves unified ``pl.<op>`` calls against
        ``pypto.language.op``, while ``inspect.signature``, IDE autocomplete and
        docstrings all show what ``pypto.language`` exports. A name bound to two
        different functions makes the parser reject arguments the visible
        signature advertises.
        """
        divergent = {
            name: (
                getattr(getattr(pl, name), "__module__", repr(getattr(pl, name))),
                getattr(getattr(language_op, name), "__module__", repr(getattr(language_op, name))),
            )
            for name in dir(language_op)
            if not name.startswith("_")
            and hasattr(pl, name)
            and getattr(pl, name) is not getattr(language_op, name)
            and callable(getattr(language_op, name))
        }
        assert not divergent, (
            f"names bound to different objects in pypto.language vs pypto.language.op: {divergent}"
        )

    @pytest.mark.parametrize("module", [pl, language_op], ids=["pl", "pl.op"])
    def test_all_lists_each_name_once(self, module):
        """``__all__`` must not repeat a name.

        ``pypto.language.op.__all__`` groups names by dispatch category
        (unified / tile-only / tensor-only). A name listed under two groups is
        invisible at runtime — ``from ... import *`` de-duplicates — but it
        makes the groups drift, and lets a later edit delete one entry while
        the name still looks unexported.
        """
        duplicates = sorted({name for name in module.__all__ if module.__all__.count(name) > 1})
        assert not duplicates, f"{module.__name__}.__all__ lists these names more than once: {duplicates}"

    def test_create_tile_single_binding(self):
        """``create_tile`` is the ``tile_ops.create`` alias in both namespaces."""
        assert pl.create_tile is language_op.create_tile is tile_ops.create
        assert "create_tile" in pl.__all__
        assert "create_tile" in language_op.__all__

    def test_promoted_create_tile_transpose(self):
        """``pl.create_tile(..., transpose=True)`` matches the explicit form.

        ``transpose=True`` is Mat-only (L1) and 2D-only — it allocates the
        transposed ZN fractal layout for a matmul ``b_trans`` B-operand.
        """

        @pl.function(type=pl.FunctionType.InCore)
        def unified(src: pl.Tensor[[256, 128], pl.BF16]) -> pl.Tensor[[256, 128], pl.BF16]:
            _t: pl.Tile[[16, 128], pl.BF16] = pl.create_tile(
                [16, 128], dtype=pl.BF16, target_memory=pl.Mem.Mat, transpose=True
            )
            return src

        @pl.function(type=pl.FunctionType.InCore)
        def explicit(src: pl.Tensor[[256, 128], pl.BF16]) -> pl.Tensor[[256, 128], pl.BF16]:
            _t: pl.Tile[[16, 128], pl.BF16] = pl.tile.create(
                [16, 128], dtype=pl.BF16, target_memory=pl.Mem.Mat, transpose=True
            )
            return src

        ir.assert_structural_equal(unified, explicit)
        # The kwarg must take effect, not merely be accepted: transpose flips
        # the sub-block layout to col_major (ZN).
        assert "slayout=pl.TileLayout.col_major" in unified.as_python()

    def test_promoted_create_tile_flat_layout(self):
        """``pl.create_tile(..., flat_layout=True)`` matches the explicit form.

        ``flat_layout`` is keyword-only and allocates a flat (non-fractal,
        ``slayout=none_box``) L1 staging buffer.
        """

        @pl.function(type=pl.FunctionType.InCore)
        def unified(src: pl.Tensor[[256, 128], pl.BF16]) -> pl.Tensor[[256, 128], pl.BF16]:
            _t: pl.Tile[[16, 128], pl.BF16] = pl.create_tile(
                [16, 128], dtype=pl.BF16, target_memory=pl.Mem.Mat, flat_layout=True
            )
            return src

        @pl.function(type=pl.FunctionType.InCore)
        def explicit(src: pl.Tensor[[256, 128], pl.BF16]) -> pl.Tensor[[256, 128], pl.BF16]:
            _t: pl.Tile[[16, 128], pl.BF16] = pl.tile.create(
                [16, 128], dtype=pl.BF16, target_memory=pl.Mem.Mat, flat_layout=True
            )
            return src

        ir.assert_structural_equal(unified, explicit)
        # The kwarg must take effect, not merely be accepted: flat_layout drops
        # the fractal sub-block boxing.
        assert "slayout=pl.TileLayout.none_box" in unified.as_python()

    def test_promoted_create(self):
        @pl.function
        def unified(a: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            c: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
            return c

        @pl.function
        def explicit(a: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            c: pl.Tensor[[64], pl.FP32] = pl.tensor.create([64], dtype=pl.FP32)
            return c

        ir.assert_structural_equal(unified, explicit)

    def test_promoted_dim(self):
        @pl.function
        def unified(a: pl.Tensor[[64, 128], pl.FP32]) -> pl.Scalar[pl.INT64]:
            d: pl.Scalar[pl.INT64] = pl.dim(a, 0)
            return d

        @pl.function
        def explicit(a: pl.Tensor[[64, 128], pl.FP32]) -> pl.Scalar[pl.INT64]:
            d: pl.Scalar[pl.INT64] = pl.tensor.dim(a, 0)
            return d

        ir.assert_structural_equal(unified, explicit)

    def test_promoted_load_store(self):
        @pl.function
        def unified(
            t: pl.Tensor[[64, 64], pl.FP32], out: pl.Tensor[[64, 64], pl.FP32]
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            a: pl.Tile[[64, 64], pl.FP32] = pl.load(t, offsets=[0, 0], shapes=[64, 64])
            result: pl.Tensor[[64, 64], pl.FP32] = pl.store(a, offsets=[0, 0], output_tensor=out)
            return result

        @pl.function
        def explicit(
            t: pl.Tensor[[64, 64], pl.FP32], out: pl.Tensor[[64, 64], pl.FP32]
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            a: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(t, offsets=[0, 0], shapes=[64, 64])
            result: pl.Tensor[[64, 64], pl.FP32] = pl.tile.store(a, offsets=[0, 0], output_tensor=out)
            return result

        ir.assert_structural_equal(unified, explicit)


class TestPromotedSinCos:
    """``pl.sin`` and ``pl.cos`` DSL wrappers (FP32-only, tensor-only)."""

    def test_pl_sin_returns_tensor(self):
        """``pl.sin(x)`` returns a ``Tensor`` wrapping a ``tensor.sin`` Call."""
        span = ir.Span.unknown()
        x = Tensor(expr=ir.Var("x", ir.TensorType([64], DataType.FP32), span))
        result = pl.sin(x)
        assert isinstance(result, Tensor)
        call = result.unwrap()
        assert isinstance(call, ir.Call)
        assert call.op.name == ir.get_op("tensor.sin").name
        result_type = call.type
        assert isinstance(result_type, ir.TensorType)
        assert result_type.dtype == DataType.FP32

    def test_pl_cos_returns_tensor(self):
        """``pl.cos(x)`` returns a ``Tensor`` wrapping a ``tensor.cos`` Call."""
        span = ir.Span.unknown()
        x = Tensor(expr=ir.Var("x", ir.TensorType([64], DataType.FP32), span))
        result = pl.cos(x)
        assert isinstance(result, Tensor)
        call = result.unwrap()
        assert isinstance(call, ir.Call)
        assert call.op.name == ir.get_op("tensor.cos").name
        result_type = call.type
        assert isinstance(result_type, ir.TensorType)
        assert result_type.dtype == DataType.FP32

    def test_pl_sin_matches_explicit(self):
        """``pl.sin`` and ``pl.tensor.sin`` produce structurally equal IR."""

        @pl.function
        def unified(a: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            c: pl.Tensor[[64], pl.FP32] = pl.sin(a)
            return c

        @pl.function
        def explicit(a: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            c: pl.Tensor[[64], pl.FP32] = pl.tensor.sin(a)
            return c

        ir.assert_structural_equal(unified, explicit)

    def test_pl_cos_matches_explicit(self):
        """``pl.cos`` and ``pl.tensor.cos`` produce structurally equal IR."""

        @pl.function
        def unified(a: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            c: pl.Tensor[[64], pl.FP32] = pl.cos(a)
            return c

        @pl.function
        def explicit(a: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            c: pl.Tensor[[64], pl.FP32] = pl.tensor.cos(a)
            return c

        ir.assert_structural_equal(unified, explicit)

    def test_pl_sin_rejects_fp16(self):
        """``pl.sin`` propagates the IR-level FP32-only validation for FP16 input."""
        span = ir.Span.unknown()
        x = Tensor(expr=ir.Var("x", ir.TensorType([64], DataType.FP16), span))
        with pytest.raises(ValueError, match=r"(?i)FP32"):
            pl.sin(x)

    def test_pl_cos_rejects_bf16(self):
        """``pl.cos`` propagates the IR-level FP32-only validation for BF16 input."""
        span = ir.Span.unknown()
        x = Tensor(expr=ir.Var("x", ir.TensorType([64], DataType.BF16), span))
        with pytest.raises(ValueError, match=r"(?i)FP32"):
            pl.cos(x)


class TestPromotedTileSinCos:
    """``pl.tile.sin`` and ``pl.tile.cos`` DSL wrappers (FP32-only, tile-only)."""

    def test_pl_tile_sin_wrapper(self):
        """``pl.tile.sin(t)`` returns a ``Tile`` wrapping a ``tile.sin`` Call."""
        span = ir.Span.unknown()
        t = Tile(expr=ir.Var("t", ir.TileType([64, 64], DataType.FP32), span))
        result = pl.tile.sin(t)
        assert isinstance(result, Tile)
        call = result.unwrap()
        assert isinstance(call, ir.Call)
        assert call.op.name == ir.get_op("tile.sin").name
        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        assert result_type.dtype == DataType.FP32

    def test_pl_tile_cos_wrapper(self):
        """``pl.tile.cos(t)`` returns a ``Tile`` wrapping a ``tile.cos`` Call."""
        span = ir.Span.unknown()
        t = Tile(expr=ir.Var("t", ir.TileType([64, 64], DataType.FP32), span))
        result = pl.tile.cos(t)
        assert isinstance(result, Tile)
        call = result.unwrap()
        assert isinstance(call, ir.Call)
        assert call.op.name == ir.get_op("tile.cos").name
        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        assert result_type.dtype == DataType.FP32

    def test_pl_tile_sin_rejects_fp16(self):
        """``pl.tile.sin`` propagates the IR-level FP32-only validation for FP16 input."""
        span = ir.Span.unknown()
        t = Tile(expr=ir.Var("t", ir.TileType([64, 64], DataType.FP16), span))
        with pytest.raises(ValueError, match=r"tile\.sin.*FP32"):
            pl.tile.sin(t)

    def test_pl_tile_cos_rejects_bf16(self):
        """``pl.tile.cos`` propagates the IR-level FP32-only validation for BF16 input."""
        span = ir.Span.unknown()
        t = Tile(expr=ir.Var("t", ir.TileType([64, 64], DataType.BF16), span))
        with pytest.raises(ValueError, match=r"tile\.cos.*FP32"):
            pl.tile.cos(t)


class TestUnifiedOpsTypeErrors:
    """Passing invalid types to unified_ops raises TypeError."""

    def test_add_invalid_lhs(self):
        with pytest.raises(TypeError, match="expected Tensor or Tile operands"):
            unified_ops.add("not_a_tensor", 1)  # type: ignore

    def test_mul_invalid_lhs(self):
        # ``pl.mul(42, 2)`` is valid scalar arithmetic — both operands are
        # ``int``, so it lowers via ``ir.mul(ConstInt(42), ConstInt(2))``
        # and returns a ``Scalar``. Reject only when a non-scalar-like
        # type slips in.
        with pytest.raises(TypeError, match="expected Tensor or Tile operands"):
            unified_ops.mul("not_a_number", 2)  # type: ignore

    def test_exp_invalid_input(self):
        with pytest.raises(TypeError, match="expected Tensor or Tile"):
            unified_ops.exp("bad")  # type: ignore

    def test_neg_invalid_input(self):
        with pytest.raises(TypeError, match="expected Tensor or Tile"):
            unified_ops.neg("bad")  # type: ignore

    def test_recip_invalid_input(self):
        with pytest.raises(TypeError, match="expected Tensor or Tile"):
            unified_ops.recip("bad")  # type: ignore

    def test_reshape_invalid_input(self):
        with pytest.raises(TypeError, match="expected Tensor or Tile"):
            unified_ops.reshape(123, [4, 4])  # type: ignore

    def test_reinterpret_view_invalid_input(self):
        with pytest.raises(TypeError, match="expected Tensor or Tile"):
            unified_ops.reinterpret_view(123, DataType.INT16)  # type: ignore

    def test_row_expand_add_rejects_tmp_for_tensor_inputs(self):
        span = ir.Span.unknown()
        lhs = Tensor(expr=ir.Var("lhs", ir.TensorType([8, 8], DataType.FP32), span))
        rhs = Tensor(expr=ir.Var("rhs", ir.TensorType([8, 1], DataType.FP32), span))
        tmp = Tile(expr=ir.Var("tmp", ir.TileType([8, 8], DataType.FP32), span))

        with pytest.raises(TypeError, match="Tensor inputs must not pass tmp"):
            unified_ops.row_expand_add(lhs, rhs, tmp)  # type: ignore[call-overload]

    def test_div_rejects_high_precision_for_scalar_paths(self):
        span = ir.Span.unknown()
        cases = [
            (
                Tensor(expr=ir.Var("tensor", ir.TensorType([8], DataType.FP32), span)),
                2.0,
            ),
            (
                Tile(expr=ir.Var("tile", ir.TileType([8], DataType.FP32), span)),
                2.0,
            ),
            (
                Scalar(expr=ir.Var("lhs", ir.ScalarType(DataType.FP32), span)),
                Scalar(expr=ir.Var("rhs", ir.ScalarType(DataType.FP32), span)),
            ),
        ]

        for lhs, rhs in cases:
            with pytest.raises(TypeError, match="high_precision"):
                unified_ops.div(lhs, rhs, high_precision=True)  # type: ignore[call-overload]

    @pytest.mark.parametrize(
        "op_name",
        [
            "row_max",
            "row_sum",
            "row_min",
            "row_prod",
            "row_argmax",
            "row_argmin",
            "col_argmax",
            "col_argmin",
        ],
    )
    def test_reduction_requires_tmp_tile_for_tile_inputs(self, op_name):
        """The scratch operand the Tensor path must omit is the one the Tile path must get.

        Both directions raise ``TypeError``: a Tile cannot synthesize caller-owned
        scratch, so omitting it is a wrong-arguments error, not a bad value.
        """
        span = ir.Span.unknown()
        tile = Tile(expr=ir.Var("input", ir.TileType([8, 64], DataType.FP32), span))

        with pytest.raises(TypeError, match="Tile inputs require tmp_tile"):
            getattr(unified_ops, op_name)(tile)

    @pytest.mark.parametrize("op_name", ["xor", "xors"])
    def test_bitwise_requires_scratch_tile_for_tile_inputs(self, op_name):
        """Same guard as the reductions, for the ops whose scratch operand is positional."""
        span = ir.Span.unknown()
        lhs = Tile(expr=ir.Var("lhs", ir.TileType([8, 64], DataType.INT32), span))

        with pytest.raises(TypeError, match="Tile inputs require an explicit scratch tile"):
            getattr(unified_ops, op_name)(lhs, 1)

    def test_cast_rejects_non_default_mode_for_scalar(self):
        """A *valid* mode the Scalar path cannot honour is a TypeError...

        ...while a mode that is not a mode at all stays a ValueError from
        ``resolve_cast_mode``, which runs first.
        """
        span = ir.Span.unknown()
        scalar = Scalar(expr=ir.Var("s", ir.ScalarType(DataType.FP32), span))

        with pytest.raises(TypeError, match="Scalar inputs do not support non-default mode"):
            unified_ops.cast(scalar, DataType.INT32, mode="floor")

        with pytest.raises(ValueError, match="Invalid rounding mode"):
            unified_ops.cast(scalar, DataType.INT32, mode="not_a_mode")

    def test_fmod_precision_dispatches_only_to_tile_tile(self):
        span = ir.Span.unknown()
        tensor = Tensor(expr=ir.Var("tensor", ir.TensorType([8], DataType.FP32), span))
        lhs = Tile(expr=ir.Var("lhs", ir.TileType([8], DataType.FP32), span))
        rhs = Tile(expr=ir.Var("rhs", ir.TileType([8], DataType.FP32), span))

        call = unified_ops.fmod(lhs, rhs, high_precision=True).unwrap()
        assert isinstance(call, ir.Call)
        assert dict(call.kwargs) == {"high_precision": True}
        with pytest.raises(TypeError, match="high_precision"):
            unified_ops.fmod(tensor, tensor, high_precision=True)  # type: ignore[call-overload]
        with pytest.raises(TypeError, match="requires a Tile rhs"):
            unified_ops.fmod(lhs, 3.0, high_precision=True)

    def test_matmul_invalid_lhs(self):
        with pytest.raises(TypeError, match="expected Tensor or Tile operands"):
            unified_ops.matmul(1, 2)  # type: ignore

    def test_add_mixed_tensor_tile(self):
        """Mixing Tensor and Tile in add gives a clear mixed-type error."""
        span = ir.Span.unknown()
        t = Tensor(expr=ir.Var("x", ir.TensorType([64], DataType.FP32), span))
        ti = Tile(expr=ir.Var("y", ir.TileType([64], DataType.FP32), span))
        with pytest.raises(TypeError, match="cannot mix Tensor and Tile"):
            unified_ops.add(t, ti)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="cannot mix Tensor and Tile"):
            unified_ops.add(ti, t)  # type: ignore[arg-type]

    def test_batch_matmul_tensor_inputs(self):
        """batch_matmul is tile-only; passing Tensors raises TypeError."""
        span = ir.Span.unknown()
        t1 = Tensor(expr=ir.Var("a", ir.TensorType([2, 64, 64], DataType.FP16), span))
        t2 = Tensor(expr=ir.Var("b", ir.TensorType([2, 64, 64], DataType.FP16), span))
        with pytest.raises(TypeError, match="expected Tensor or Tile operands"):
            unified_ops.batch_matmul(t1, t2)  # type: ignore[arg-type]

    def test_batch_matmul_invalid_lhs(self):
        """batch_matmul with non-Tensor/Tile input raises TypeError."""
        with pytest.raises(TypeError, match="expected Tensor or Tile operands"):
            unified_ops.batch_matmul(1, 2)  # type: ignore


# The unified wrappers accept the union of both levels' kwargs. A kwarg only the
# *other* dispatch path can honour must raise instead of being dropped — a
# discarded ``b_trans`` compiles wrong math, and a discarded scratch tile leaves
# the caller's buffer dead while still consuming UB budget. Only a non-default
# value raises; spelling out the documented default keeps working.
_TMP_TILE_REDUCTIONS = [
    "row_max",
    "row_sum",
    "row_min",
    "row_prod",
    "col_sum",
    "row_argmax",
    "row_argmin",
    "col_argmax",
    "col_argmin",
]


def _tile(name: str, shape: list[int], dtype: DataType = DataType.FP16) -> Tile:
    return Tile(expr=ir.Var(name, ir.TileType(shape, dtype), ir.Span.unknown()))


def _tensor(name: str, shape: list[int], dtype: DataType = DataType.FP32) -> Tensor:
    return Tensor(expr=ir.Var(name, ir.TensorType(shape, dtype), ir.Span.unknown()))


class TestUnifiedOpsCrossPathKwargs:
    """Kwargs only one dispatch path can honour raise instead of being dropped."""

    @pytest.mark.parametrize(
        "kwarg,remedy",
        [
            ("a_trans", "transpose_view"),
            ("b_trans", "transpose_view"),
            ("c_matrix_nz", "Acc tile type"),
        ],
    )
    def test_matmul_tile_rejects_tensor_only_flags(self, kwarg, remedy):
        """Tensor-level matmul flags have no tile equivalent, so they must raise."""
        lhs, rhs = _tile("lhs", [32, 128]), _tile("rhs", [128, 128])
        with pytest.raises(TypeError) as exc_info:
            unified_ops.matmul(lhs, rhs, **{kwarg: True})  # type: ignore[call-overload]

        msg = str(exc_info.value)
        assert f"'{kwarg}'" in msg
        assert "not supported for Tile operands" in msg
        assert remedy in msg

    def test_matmul_tile_accepts_explicit_default_flags(self):
        """Spelling out the defaults is a no-op and yields plain pl.tile.matmul IR."""
        lhs, rhs = _tile("lhs", [32, 128]), _tile("rhs", [128, 128])

        unified = unified_ops.matmul(lhs, rhs, a_trans=False, b_trans=False, c_matrix_nz=False)
        explicit = pl.tile.matmul(lhs, rhs)

        ir.assert_structural_equal(unified.unwrap(), explicit.unwrap())

    def test_matmul_tile_accepts_out_dtype_matching_deduction(self):
        """tile.matmul deduces FP32 for float operands; asking for FP32 is honoured."""
        lhs, rhs = _tile("lhs", [32, 128]), _tile("rhs", [128, 128])

        unified = unified_ops.matmul(lhs, rhs, out_dtype=DataType.FP32)
        explicit = pl.tile.matmul(lhs, rhs)

        result_type = unified.unwrap().type
        assert isinstance(result_type, ir.TileType)
        assert result_type.dtype == DataType.FP32
        ir.assert_structural_equal(unified.unwrap(), explicit.unwrap())

    def test_matmul_tile_rejects_out_dtype_the_accumulator_cannot_produce(self):
        """The Cube accumulator is fixed at FP32 here, so FP16 must raise, not drop."""
        lhs, rhs = _tile("lhs", [32, 128]), _tile("rhs", [128, 128])
        with pytest.raises(TypeError) as exc_info:
            unified_ops.matmul(lhs, rhs, out_dtype=DataType.FP16)

        msg = str(exc_info.value)
        assert "out_dtype" in msg
        assert "fp32" in msg  # names what it actually deduced
        assert "pl.cast" in msg

    def test_matmul_tile_rejects_unverifiable_int_out_dtype(self):
        """A raw int dtype code cannot be compared against the deduction, so it raises.

        ``DataType`` exposes no Python int conversion, so an int value cannot be
        checked against what tile.matmul actually deduced — and skipping the
        check is the silent drop this guard exists to prevent. The Tile overload
        already rejects this statically (hence the suppression); the runtime
        check still matters because the DSL parser reaches the wrapper
        dynamically.
        """
        lhs, rhs = _tile("lhs", [32, 128]), _tile("rhs", [128, 128])
        with pytest.raises(TypeError, match="out_dtype"):
            unified_ops.matmul(lhs, rhs, out_dtype=51)  # pyright: ignore[reportArgumentType]

    def test_matmul_tensor_still_honors_all_kwargs(self):
        """The Tensor path is untouched — every kwarg still reaches tensor.matmul."""
        # b_trans=True means rhs is [N, K], so [512, 128] against an lhs K of 128.
        lhs, rhs = _tensor("lhs", [32, 128], DataType.BF16), _tensor("rhs", [512, 128], DataType.BF16)

        unified = unified_ops.matmul(lhs, rhs, out_dtype=DataType.FP32, a_trans=False, b_trans=True)
        explicit = pl.tensor.matmul(lhs, rhs, DataType.FP32, False, True)

        ir.assert_structural_equal(unified.unwrap(), explicit.unwrap())

    @pytest.mark.parametrize("kwarg", ["a_trans", "b_trans"])
    def test_matmul_acc_tile_rejects_transpose_flags(self, kwarg):
        acc = _tile("acc", [32, 128], DataType.FP32)
        lhs, rhs = _tile("lhs", [32, 128]), _tile("rhs", [128, 128])
        with pytest.raises(TypeError) as exc_info:
            unified_ops.matmul_acc(acc, lhs, rhs, **{kwarg: True})  # type: ignore[call-overload]

        msg = str(exc_info.value)
        assert f"'{kwarg}'" in msg
        assert "transpose_view" in msg

    def test_matmul_acc_tile_accepts_explicit_default_flags(self):
        acc = _tile("acc", [32, 128], DataType.FP32)
        lhs, rhs = _tile("lhs", [32, 128]), _tile("rhs", [128, 128])

        unified = unified_ops.matmul_acc(acc, lhs, rhs, a_trans=False, b_trans=False)
        explicit = pl.tile.matmul_acc(acc, lhs, rhs)

        ir.assert_structural_equal(unified.unwrap(), explicit.unwrap())

    def test_rsqrt_tile_rejects_high_precision(self):
        """tile.rsqrt selects precision by taking a scratch tile, not by a flag."""
        with pytest.raises(TypeError) as exc_info:
            unified_ops.rsqrt(_tile("t", [64, 64], DataType.FP32), high_precision=True)  # type: ignore[call-overload]

        msg = str(exc_info.value)
        assert "'high_precision'" in msg
        assert "pl.tile.rsqrt(tile, tmp)" in msg

    def test_rsqrt_tile_default_still_lowers(self):
        t = _tile("t", [64, 64], DataType.FP32)

        unified = unified_ops.rsqrt(t)
        explicit = pl.tile.rsqrt(t)

        ir.assert_structural_equal(unified.unwrap(), explicit.unwrap())

    def test_rsqrt_tile_accepts_explicit_default_high_precision(self):
        """Spelling out the default is a no-op the overloads must also accept."""
        t = _tile("t", [64, 64], DataType.FP32)

        unified = unified_ops.rsqrt(t, high_precision=False)
        explicit = pl.tile.rsqrt(t)

        ir.assert_structural_equal(unified.unwrap(), explicit.unwrap())

    def test_rsqrt_tensor_still_honors_high_precision(self):
        x = _tensor("x", [64, 64])

        unified = unified_ops.rsqrt(x, high_precision=True)
        explicit = pl.tensor.rsqrt(x, high_precision=True)

        ir.assert_structural_equal(unified.unwrap(), explicit.unwrap())

    @pytest.mark.parametrize("op_name", _TMP_TILE_REDUCTIONS)
    def test_reduction_tensor_path_rejects_tmp_tile(self, op_name):
        """The conversion pass allocates the scratch, so a user tmp_tile must raise."""
        x = _tensor("x", [64, 64])
        tmp = _tile("tmp", [64, 64], DataType.FP32)
        with pytest.raises(TypeError) as exc_info:
            getattr(unified_ops, op_name)(x, tmp)  # type: ignore[call-overload]

        msg = str(exc_info.value)
        assert f"pl.{op_name}" in msg
        assert "tmp_tile" in msg

    @pytest.mark.parametrize("op_name", _TMP_TILE_REDUCTIONS)
    def test_reduction_tensor_path_without_tmp_tile_unchanged(self, op_name):
        """Omitting tmp_tile on the Tensor path still matches pl.tensor.<op>."""
        x = _tensor("x", [64, 64])

        unified = getattr(unified_ops, op_name)(x)
        explicit = getattr(pl.tensor, op_name)(x)

        ir.assert_structural_equal(unified.unwrap(), explicit.unwrap())

    @pytest.mark.parametrize("op_name", _TMP_TILE_REDUCTIONS)
    def test_reduction_tensor_path_accepts_explicit_none_tmp_tile(self, op_name):
        """Passing the default explicitly stays legal — only a real tile raises."""
        x = _tensor("x", [64, 64])

        unified = getattr(unified_ops, op_name)(x, None)
        explicit = getattr(pl.tensor, op_name)(x)

        ir.assert_structural_equal(unified.unwrap(), explicit.unwrap())

    def test_col_sum_tile_path_still_selects_binary_tree(self):
        """tmp_tile is honoured on the Tile path — it selects binary-tree reduction."""
        t = _tile("t", [64, 64], DataType.FP32)
        tmp = _tile("tmp", [64, 64], DataType.FP32)

        unified = unified_ops.col_sum(t, tmp)
        explicit = pl.tile.col_sum(t, tmp)

        ir.assert_structural_equal(unified.unwrap(), explicit.unwrap())
        # The binary-tree form is distinguishable from the sequential one.
        assert not ir.structural_equal(unified.unwrap(), pl.tile.col_sum(t).unwrap())


class TestUnifiedSlicePadValue:
    """``pl.slice`` forwards ``pad_value``, which both dispatch paths honour.

    ``clamp`` is the only slice kwarg one path cannot honour; ``pad_value``
    exists at both levels, so it is plain forwarding rather than a cross-path
    guard. Each test pairs the structural match against the explicit call with
    a negative assertion that the no-``pad_value`` IR differs — otherwise the
    match would stay green if both sides dropped the kwarg.
    """

    def _pad_of(self, value: Tensor | Tile) -> ir.PadValue:
        """Padding mode recorded on a sliced value's view."""
        value_type = value.unwrap().type
        view = getattr(value_type, "tensor_view", None)
        if view is None:
            view = getattr(value_type, "tile_view", None)
        assert view is not None, "a narrowed slice must carry a view"
        return view.pad

    def test_tensor_path_forwards_pad_value(self):
        x = _tensor("x", [16, 32])

        unified = unified_ops.slice(x, [8, 32], [0, 0], valid_shape=[8, 8], pad_value=ir.PadValue.min)
        explicit = pl.tensor.slice(x, [8, 32], [0, 0], valid_shape=[8, 8], pad_value=ir.PadValue.min)

        ir.assert_structural_equal(unified.unwrap(), explicit.unwrap())
        assert self._pad_of(unified) == ir.PadValue.min
        unpadded = pl.tensor.slice(x, [8, 32], [0, 0], valid_shape=[8, 8])
        assert not ir.structural_equal(unified.unwrap(), unpadded.unwrap())

    def test_tile_path_forwards_pad_value(self):
        t = _tile("t", [16, 32])

        unified = unified_ops.slice(t, [8, 32], [0, 0], valid_shape=[8, 8], pad_value=ir.PadValue.min)
        explicit = pl.tile.slice(t, [8, 32], [0, 0], valid_shape=[8, 8], pad_value=ir.PadValue.min)

        ir.assert_structural_equal(unified.unwrap(), explicit.unwrap())
        assert self._pad_of(unified) == ir.PadValue.min
        unpadded = pl.tile.slice(t, [8, 32], [0, 0], valid_shape=[8, 8])
        assert not ir.structural_equal(unified.unwrap(), unpadded.unwrap())

    def test_literal_sugar_forwards(self):
        """The ``0`` / ``inf`` sugars resolve on the unified path too."""
        x = _tensor("x", [16, 32])

        unified = unified_ops.slice(x, [8, 32], [0, 0], valid_shape=[8, 8], pad_value=0)
        explicit = pl.tensor.slice(x, [8, 32], [0, 0], valid_shape=[8, 8], pad_value=ir.PadValue.zero)

        ir.assert_structural_equal(unified.unwrap(), explicit.unwrap())
        assert self._pad_of(unified) == ir.PadValue.zero

    def test_pad_value_binds_sixth_positional(self):
        """Positional order matches ``pl.tensor.slice`` — pad_value precedes clamp."""
        x = _tensor("x", [16, 32])

        unified = unified_ops.slice(x, [8, 32], [0, 0], [8, 8], None, ir.PadValue.max)
        explicit = pl.tensor.slice(x, [8, 32], [0, 0], [8, 8], None, ir.PadValue.max)

        ir.assert_structural_equal(unified.unwrap(), explicit.unwrap())
        assert self._pad_of(unified) == ir.PadValue.max

    def test_pad_value_rides_alongside_clamp(self):
        """clamp narrows the valid region, so pad_value has something to paint."""
        x = _tensor("x", [16, 32])

        unified = unified_ops.slice(x, [16, 32], [8, 0], pad_value=ir.PadValue.max, clamp=True)
        explicit = pl.tensor.slice(x, [16, 32], [8, 0], pad_value=ir.PadValue.max, clamp=True)

        ir.assert_structural_equal(unified.unwrap(), explicit.unwrap())
        assert self._pad_of(unified) == ir.PadValue.max

    @pytest.mark.parametrize("kind", ["tensor", "tile"])
    def test_omitted_pad_value_is_unchanged(self, kind):
        """The default stays byte-identical to the pre-``pad_value`` behaviour."""
        # One shared source Var — structural equality matches Vars by pointer.
        if kind == "tensor":
            src: Tensor | Tile = _tensor("x", [16, 32])
            explicit = pl.tensor.slice(src, [8, 32], [0, 0], valid_shape=[8, 8])
        else:
            src = _tile("t", [16, 32])
            explicit = pl.tile.slice(src, [8, 32], [0, 0], valid_shape=[8, 8])

        unified = unified_ops.slice(src, [8, 32], [0, 0], valid_shape=[8, 8])
        with_none = unified_ops.slice(src, [8, 32], [0, 0], valid_shape=[8, 8], pad_value=None)

        ir.assert_structural_equal(unified.unwrap(), explicit.unwrap())
        ir.assert_structural_equal(with_none.unwrap(), explicit.unwrap())

    @pytest.mark.parametrize("kind", ["tensor", "tile"])
    def test_pad_value_without_narrowing_still_warns(self, kind):
        """The underlying no-op warning is not swallowed by the dispatcher."""
        src: Tensor | Tile = _tensor("x", [16, 32]) if kind == "tensor" else _tile("t", [16, 32])

        with pytest.warns(UserWarning, match="pad_value has no effect"):
            unified_ops.slice(src, [8, 32], [0, 0], pad_value=ir.PadValue.min)

    @pytest.mark.parametrize("kind", ["tensor", "tile"])
    def test_warning_names_the_caller_not_the_dispatcher(self, kind):
        """The warning must point at user code, not at ``unified_ops``.

        ``pytest.warns`` checks neither filename nor lineno, so it cannot catch
        this; the assertion has to read the record.
        """
        src: Tensor | Tile = _tensor("x", [16, 32]) if kind == "tensor" else _tile("t", [16, 32])

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            unified_ops.slice(src, [8, 32], [0, 0], pad_value=ir.PadValue.min)

        assert len(caught) == 1
        assert caught[0].filename == __file__

    @pytest.mark.parametrize("kind", ["tensor", "tile"])
    def test_each_call_site_warns_under_the_default_filter(self, kind):
        """Distinct call sites must not collapse into a single warning.

        The default filter dedupes on ``(text, category, lineno)`` in the frame
        named by ``stacklevel``. A dispatcher-fixed stacklevel would key every
        call site to one library line, showing the first and silently dropping
        the rest.
        """
        src: Tensor | Tile = _tensor("x", [16, 32]) if kind == "tensor" else _tile("t", [16, 32])

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("default")
            unified_ops.slice(src, [8, 32], [0, 0], pad_value=ir.PadValue.min)
            unified_ops.slice(src, [8, 32], [0, 0], pad_value=ir.PadValue.min)
            unified_ops.slice(src, [8, 32], [0, 0], pad_value=ir.PadValue.min)

        assert len(caught) == 3, "each distinct call site must surface its own warning"

    def test_pad_value_reaches_the_parser_path(self):
        """A DSL body reaches the wrapper dynamically — pad_value must survive that."""

        @pl.function
        def unified(x: pl.Tensor[[16, 32], pl.FP32]) -> pl.Tensor[[16, 32], pl.FP32]:
            narrowed = pl.slice(x, [8, 32], [0, 0], valid_shape=[8, 8], pad_value=pl.PadValue.min)
            return pl.fillpad_expand(narrowed, [16, 32])

        @pl.function
        def explicit(x: pl.Tensor[[16, 32], pl.FP32]) -> pl.Tensor[[16, 32], pl.FP32]:
            narrowed = pl.tensor.slice(x, [8, 32], [0, 0], valid_shape=[8, 8], pad_value=pl.PadValue.min)
            return pl.fillpad_expand(narrowed, [16, 32])

        ir.assert_structural_equal(unified, explicit)

        @pl.function
        def unpadded(x: pl.Tensor[[16, 32], pl.FP32]) -> pl.Tensor[[16, 32], pl.FP32]:
            narrowed = pl.tensor.slice(x, [8, 32], [0, 0], valid_shape=[8, 8])
            return pl.fillpad_expand(narrowed, [16, 32])

        assert not ir.structural_equal(unified, unpadded)

    def test_sibling_directory_is_not_mistaken_for_library_code(self):
        """A user path that merely *starts with* the package dir is user code.

        ``<parent>/pypto_kernels/k.py`` prefix-matches ``<parent>/pypto``, so a
        bare ``startswith`` on the package directory would skip a real user
        frame and walk on to name someone else's. The match has to be on a path
        component.
        """
        pkg_dir = pathlib.Path(pypto_ir_utils.__file__).resolve().parent.parent
        sibling = f"{pkg_dir}_kernels{os.sep}k.py"

        namespace = {"probe": pypto_ir_utils.caller_warning_stacklevel}
        exec(compile("def warn_site():\n    return probe()\n", sibling, "exec"), namespace)  # noqa: S102

        # The frame that would call warnings.warn is itself user code, so the
        # level naming it is 1 — not 2, which would name this test instead.
        assert namespace["warn_site"]() == 1

    def test_pad_value_reaches_the_parser_path_for_tiles(self):
        """The Tile forward is reached dynamically through the parser too."""

        @pl.function
        def unified(x: pl.Tensor[[16, 32], pl.FP32]) -> pl.Tensor[[16, 32], pl.FP32]:
            t = pl.load(x, [0, 0], [16, 32], target_memory=pl.MemorySpace.Vec)
            narrowed = pl.slice(t, [8, 32], [0, 0], valid_shape=[8, 8], pad_value=pl.PadValue.min)
            return pl.store(pl.fillpad_expand(narrowed, [16, 32]), [0, 0], x)

        @pl.function
        def explicit(x: pl.Tensor[[16, 32], pl.FP32]) -> pl.Tensor[[16, 32], pl.FP32]:
            t = pl.load(x, [0, 0], [16, 32], target_memory=pl.MemorySpace.Vec)
            narrowed = pl.tile.slice(t, [8, 32], [0, 0], valid_shape=[8, 8], pad_value=pl.PadValue.min)
            return pl.store(pl.fillpad_expand(narrowed, [16, 32]), [0, 0], x)

        ir.assert_structural_equal(unified, explicit)

        @pl.function
        def unpadded(x: pl.Tensor[[16, 32], pl.FP32]) -> pl.Tensor[[16, 32], pl.FP32]:
            t = pl.load(x, [0, 0], [16, 32], target_memory=pl.MemorySpace.Vec)
            narrowed = pl.tile.slice(t, [8, 32], [0, 0], valid_shape=[8, 8])
            return pl.store(pl.fillpad_expand(narrowed, [16, 32]), [0, 0], x)

        assert not ir.structural_equal(unified, unpadded)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
