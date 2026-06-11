# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for tile.interleave / tile.deinterleave (issue #1325).

Type contract (enforced by the shared op type deduction):
    * ``lhs`` and ``rhs`` must be tiles with identical dtype, shape, and
      valid_shape; tiles live in Vec.
    * dtype element width must be 8/16/32-bit.
    * Both outputs copy the lhs tile type (same dtype/shape/valid_shape).
    * Result is an ordered pair: interleave -> (low, high),
      deinterleave -> (even, odd).
"""

import pypto.language as pl
import pytest

_VALID_DTYPES = [pl.INT8, pl.UINT8, pl.FP16, pl.BF16, pl.INT16, pl.FP32, pl.INT32, pl.UINT32]
_INVALID_DTYPES = [pl.INT64]


def _build_interleave_program(dtype=pl.FP32):
    @pl.program
    class ProgInterleave:
        @pl.function(type=pl.FunctionType.InCore)
        def main(
            self,
            lhs: pl.Tensor[[32, 64], dtype],
            rhs: pl.Tensor[[32, 64], dtype],
            out_low: pl.Tensor[[32, 64], dtype],
            out_high: pl.Tensor[[32, 64], dtype],
        ):
            a: pl.Tile[[32, 64], dtype] = pl.load(lhs, [0, 0], [32, 64])
            b: pl.Tile[[32, 64], dtype] = pl.load(rhs, [0, 0], [32, 64])
            low, high = pl.tile.interleave(a, b)
            pl.store(low, [0, 0], out_low)
            pl.store(high, [0, 0], out_high)

    return ProgInterleave


def _build_deinterleave_program(dtype=pl.FP32):
    @pl.program
    class ProgDeinterleave:
        @pl.function(type=pl.FunctionType.InCore)
        def main(
            self,
            lhs: pl.Tensor[[32, 64], dtype],
            rhs: pl.Tensor[[32, 64], dtype],
            out_even: pl.Tensor[[32, 64], dtype],
            out_odd: pl.Tensor[[32, 64], dtype],
        ):
            a: pl.Tile[[32, 64], dtype] = pl.load(lhs, [0, 0], [32, 64])
            b: pl.Tile[[32, 64], dtype] = pl.load(rhs, [0, 0], [32, 64])
            even, odd = pl.tile.deinterleave(a, b)
            pl.store(even, [0, 0], out_even)
            pl.store(odd, [0, 0], out_odd)

    return ProgDeinterleave


class TestTileInterleaveTypes:
    """Type-contract tests for tile.interleave: outputs mirror the lhs type."""

    @pytest.mark.parametrize("dtype", _VALID_DTYPES)
    def test_valid_dtype(self, dtype):
        prog = _build_interleave_program(dtype=dtype)
        text = str(prog)
        assert "tile.interleave" in text
        # Both outputs keep the input dtype/shape: stores into [32, 64] tensors
        # of the same dtype type-check, so the op call must be present.
        assert text.count("tile.interleave") == 1

    @pytest.mark.parametrize("dtype", _INVALID_DTYPES)
    def test_invalid_dtype_raises(self, dtype):
        with pytest.raises(Exception, match="8/16/32-bit"):
            _build_interleave_program(dtype=dtype)

    def test_dtype_mismatch_raises(self):
        with pytest.raises(Exception, match="dtype to match"):

            @pl.program
            class BadDtype:
                @pl.function(type=pl.FunctionType.InCore)
                def main(
                    self,
                    lhs: pl.Tensor[[32, 64], pl.FP32],
                    rhs: pl.Tensor[[32, 64], pl.FP16],
                    out_low: pl.Tensor[[32, 64], pl.FP32],
                    out_high: pl.Tensor[[32, 64], pl.FP32],
                ):
                    a: pl.Tile[[32, 64], pl.FP32] = pl.load(lhs, [0, 0], [32, 64])
                    b: pl.Tile[[32, 64], pl.FP16] = pl.load(rhs, [0, 0], [32, 64])
                    low, high = pl.tile.interleave(a, b)
                    pl.store(low, [0, 0], out_low)
                    pl.store(high, [0, 0], out_high)

    def test_shape_mismatch_raises(self):
        with pytest.raises(Exception, match="shapes to match"):

            @pl.program
            class BadShape:
                @pl.function(type=pl.FunctionType.InCore)
                def main(
                    self,
                    lhs: pl.Tensor[[32, 64], pl.FP32],
                    rhs: pl.Tensor[[32, 32], pl.FP32],
                    out_low: pl.Tensor[[32, 64], pl.FP32],
                    out_high: pl.Tensor[[32, 64], pl.FP32],
                ):
                    a: pl.Tile[[32, 64], pl.FP32] = pl.load(lhs, [0, 0], [32, 64])
                    b: pl.Tile[[32, 32], pl.FP32] = pl.load(rhs, [0, 0], [32, 32])
                    low, high = pl.tile.interleave(a, b)
                    pl.store(low, [0, 0], out_low)
                    pl.store(high, [0, 0], out_high)

    def test_non_2d_raises(self):
        with pytest.raises(Exception, match="2D tiles"):

            @pl.program
            class Bad3D:
                @pl.function(type=pl.FunctionType.InCore)
                def main(
                    self,
                    lhs: pl.Tensor[[2, 32, 64], pl.FP32],
                    rhs: pl.Tensor[[2, 32, 64], pl.FP32],
                    out_low: pl.Tensor[[2, 32, 64], pl.FP32],
                    out_high: pl.Tensor[[2, 32, 64], pl.FP32],
                ):
                    a: pl.Tile[[2, 32, 64], pl.FP32] = pl.load(lhs, [0, 0, 0], [2, 32, 64])
                    b: pl.Tile[[2, 32, 64], pl.FP32] = pl.load(rhs, [0, 0, 0], [2, 32, 64])
                    low, high = pl.tile.interleave(a, b)
                    pl.store(low, [0, 0, 0], out_low)
                    pl.store(high, [0, 0, 0], out_high)

    def test_valid_shape_mismatch_raises(self):
        with pytest.raises(Exception, match="valid_shape to match"):

            @pl.program
            class BadValidShape:
                @pl.function(type=pl.FunctionType.InCore)
                def main(
                    self,
                    lhs: pl.Tensor[[32, 64], pl.FP32],
                    rhs: pl.Tensor[[32, 64], pl.FP32],
                    out_low: pl.Tensor[[32, 64], pl.FP32],
                    out_high: pl.Tensor[[32, 64], pl.FP32],
                ):
                    a: pl.Tile[[32, 64], pl.FP32] = pl.load(lhs, [0, 0], [32, 64])
                    b_full: pl.Tile[[32, 64], pl.FP32] = pl.load(rhs, [0, 0], [32, 64])
                    b: pl.Tile[[32, 64], pl.FP32] = pl.tile.set_validshape(b_full, 32, 32)
                    low, high = pl.tile.interleave(a, b)
                    pl.store(low, [0, 0], out_low)
                    pl.store(high, [0, 0], out_high)


class TestTileDeinterleaveTypes:
    """Type-contract tests for tile.deinterleave: same contract as interleave."""

    @pytest.mark.parametrize("dtype", _VALID_DTYPES)
    def test_valid_dtype(self, dtype):
        prog = _build_deinterleave_program(dtype=dtype)
        text = str(prog)
        assert "tile.deinterleave" in text
        assert text.count("tile.deinterleave") == 1

    @pytest.mark.parametrize("dtype", _INVALID_DTYPES)
    def test_invalid_dtype_raises(self, dtype):
        with pytest.raises(Exception, match="8/16/32-bit"):
            _build_deinterleave_program(dtype=dtype)

    def test_dtype_mismatch_raises(self):
        with pytest.raises(Exception, match="dtype to match"):

            @pl.program
            class BadDtypeDeintlv:
                @pl.function(type=pl.FunctionType.InCore)
                def main(
                    self,
                    lhs: pl.Tensor[[32, 64], pl.INT16],
                    rhs: pl.Tensor[[32, 64], pl.INT32],
                    out_even: pl.Tensor[[32, 64], pl.INT16],
                    out_odd: pl.Tensor[[32, 64], pl.INT16],
                ):
                    a: pl.Tile[[32, 64], pl.INT16] = pl.load(lhs, [0, 0], [32, 64])
                    b: pl.Tile[[32, 64], pl.INT32] = pl.load(rhs, [0, 0], [32, 64])
                    even, odd = pl.tile.deinterleave(a, b)
                    pl.store(even, [0, 0], out_even)
                    pl.store(odd, [0, 0], out_odd)

    def test_shape_mismatch_raises(self):
        with pytest.raises(Exception, match="shapes to match"):

            @pl.program
            class BadShapeDeintlv:
                @pl.function(type=pl.FunctionType.InCore)
                def main(
                    self,
                    lhs: pl.Tensor[[32, 64], pl.FP32],
                    rhs: pl.Tensor[[16, 64], pl.FP32],
                    out_even: pl.Tensor[[32, 64], pl.FP32],
                    out_odd: pl.Tensor[[32, 64], pl.FP32],
                ):
                    a: pl.Tile[[32, 64], pl.FP32] = pl.load(lhs, [0, 0], [32, 64])
                    b: pl.Tile[[16, 64], pl.FP32] = pl.load(rhs, [0, 0], [16, 64])
                    even, odd = pl.tile.deinterleave(a, b)
                    pl.store(even, [0, 0], out_even)
                    pl.store(odd, [0, 0], out_odd)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
