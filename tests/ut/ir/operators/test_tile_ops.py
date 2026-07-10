# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for tile operations."""

import math

import pypto.language as pl
import pytest
from pypto import DataType, ir
from pypto.ir.op import tensor, tile


def _mk_tile(name, shape, valid=None, dtype=DataType.FP32, blayout=None):
    """Build a tile-typed Var with an optional explicit per-axis ``valid_shape``.

    ``shape`` entries are static ints. ``valid`` entries may be ints (static extents)
    or ``Expr`` nodes (symbolic/dynamic extents); ``None`` builds a fully-valid tile
    (no explicit ``TileView``, so ``GetValidShape`` falls back to the physical shape).
    ``blayout`` optionally stamps the tile's block layout (used by the elementwise
    valid-shape agreement tests to prove the result layout comes from the shaping
    operand, not a shape-1 broadcast operand).
    """
    span = ir.Span.unknown()
    shape_exprs = [ir.ConstInt(s, DataType.INT32, span) for s in shape]
    if valid is None and blayout is None:
        view = None
    else:
        kwargs = {}
        if valid is not None:
            kwargs["valid_shape"] = list(valid)
        if blayout is not None:
            kwargs["blayout"] = blayout
        view = ir.TileView(**kwargs)
    return ir.Var(name, ir.TileType(shape_exprs, dtype, None, view), span)


def _index_scalar(name):
    """A symbolic INDEX-typed scalar Var usable as a dynamic valid extent."""
    return ir.Var(name, ir.ScalarType(DataType.INDEX), ir.Span.unknown())


def _effective_valid(tile_type):
    """Effective ``valid_shape`` of a tile type (physical shape when unset)."""
    view = tile_type.tile_view
    if view is None or len(view.valid_shape) == 0:
        return list(tile_type.shape)
    return list(view.valid_shape)


def _valid_ints(tile_type):
    """Effective ``valid_shape`` as a list of ints; asserts every extent is static."""
    out = []
    for extent in _effective_valid(tile_type):
        assert isinstance(extent, ir.ConstInt), f"expected static valid extent, got {extent!r}"
        out.append(extent.value)
    return out


def _const_ints(exprs) -> list[int]:
    """Extract ``.value`` from a sequence of ConstInt exprs; asserts each is static."""
    out = []
    for e in exprs:
        assert isinstance(e, ir.ConstInt)
        out.append(e.value)
    return out


class TestTileElementwiseOps:
    """Test suite for tile-level element-wise operators (tile-tile and tile-scalar)."""

    def test_tile_add(self):
        """Test tile.add operator - element-wise addition of two tiles."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                b: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(a, [0, 0], [32, 32])
                tile_b: pl.Tile[[32, 32], pl.FP32] = pl.load(b, [0, 0], [32, 32])
                tile_c: pl.Tile[[32, 32], pl.FP32] = pl.add(tile_a, tile_b)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.add" in ir_str

    def test_tile_sub(self):
        """Test tile.sub operator - element-wise subtraction of two tiles."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                b: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(a, [0, 0], [32, 32])
                tile_b: pl.Tile[[32, 32], pl.FP32] = pl.load(b, [0, 0], [32, 32])
                tile_c: pl.Tile[[32, 32], pl.FP32] = pl.sub(tile_a, tile_b)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.sub" in ir_str

    def test_tile_mul(self):
        """Test tile.mul operator - element-wise multiplication of two tiles."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                b: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(a, [0, 0], [32, 32])
                tile_b: pl.Tile[[32, 32], pl.FP32] = pl.load(b, [0, 0], [32, 32])
                tile_c: pl.Tile[[32, 32], pl.FP32] = pl.mul(tile_a, tile_b)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.mul" in ir_str

    def test_tile_div(self):
        """Test tile.div operator - element-wise division of two tiles."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                b: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(a, [0, 0], [32, 32])
                tile_b: pl.Tile[[32, 32], pl.FP32] = pl.load(b, [0, 0], [32, 32])
                tile_c: pl.Tile[[32, 32], pl.FP32] = pl.div(tile_a, tile_b)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.div" in ir_str

    def test_tile_muls(self):
        """Test tile.muls operator - multiply all elements of a tile by scalar."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(a, [0, 0], [32, 32])
                tile_c: pl.Tile[[32, 32], pl.FP32] = pl.mul(tile_a, 2.0)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.muls" in ir_str

    def test_tile_muls_preserves_tile_dtype(self):
        """tile.muls result must keep the tile's element dtype, not promote to the scalar's dtype.

        pto.tmuls requires src and dst to share the same element type, so multiplying a BF16
        tile by an FP32 scalar must produce a BF16 result (the scalar is narrowed at runtime).
        """

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.BF16],
                output: pl.Tensor[[128, 128], pl.BF16],
            ) -> pl.Tensor[[128, 128], pl.BF16]:
                tile_a: pl.Tile[[32, 32], pl.BF16] = pl.load(a, [0, 0], [32, 32])
                # Scalar 0.0 is typed FP32 by default; result must still be BF16.
                tile_c: pl.Tile[[32, 32], pl.BF16] = pl.mul(tile_a, 0.0)
                result: pl.Tensor[[128, 128], pl.BF16] = pl.store(tile_c, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.muls" in ir_str
        # Confirm the result tile carries BF16 (pl.BF16 in the Python printer),
        # not a promoted FP32.  The hardware narrowing happens at runtime.
        assert "tile_c: pl.Tile[[32, 32], pl.BF16" in ir_str

    def test_tile_cmp(self):
        """Test tile.cmp operator - element-wise comparison of two tiles."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                b: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(a, [0, 0], [32, 32])
                tile_b: pl.Tile[[32, 32], pl.FP32] = pl.load(b, [0, 0], [32, 32])
                tile_c: pl.Tile[[32, 32], pl.UINT8] = pl.cmp(tile_a, tile_b, cmp_type=0)
                one_tile: pl.Tile[[32, 32], pl.FP32] = pl.tile.full([32, 32], dtype=pl.FP32, value=1.0)
                zero_tile: pl.Tile[[32, 32], pl.FP32] = pl.tile.full([32, 32], dtype=pl.FP32, value=0.0)
                tmp: pl.Tile[[1, 32], pl.UINT8] = pl.tile.create([1, 32], dtype=pl.UINT8)
                selected: pl.Tile[[32, 32], pl.FP32] = pl.sel(tile_c, one_tile, zero_tile, tmp)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(selected, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.cmp" in ir_str

    def test_tile_cmps(self):
        """Test tile.cmps operator - compare tile elements with scalar."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(a, [0, 0], [32, 32])
                tile_c: pl.Tile[[32, 32], pl.UINT8] = pl.cmps(tile_a, 0.0, cmp_type=0)
                one_tile: pl.Tile[[32, 32], pl.FP32] = pl.tile.full([32, 32], dtype=pl.FP32, value=1.0)
                zero_tile: pl.Tile[[32, 32], pl.FP32] = pl.tile.full([32, 32], dtype=pl.FP32, value=0.0)
                tmp: pl.Tile[[1, 32], pl.UINT8] = pl.tile.create([1, 32], dtype=pl.UINT8)
                selected: pl.Tile[[32, 32], pl.FP32] = pl.sel(tile_c, one_tile, zero_tile, tmp)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(selected, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.cmps" in ir_str


def _blayout(tile_type):
    """Effective block layout of a tile type (resolves the canonicalized view)."""
    return tile_type.get_effective_tile_view().blayout


class TestTileElementwiseValidShapeAgreement:
    """4e: multi-operand tile ops AGREE on their operands' valid regions instead of
    blindly taking arg0's (valid_shape never broadcasts)."""

    def test_add_commutative_valid_shape_and_layout(self):
        """pl.add(at, bt) and pl.add(bt, at) infer the SAME valid_shape and SAME blayout.

        This is the motivating r2 bug: arg-order used to change both the valid_shape
        (narrowing to the [128,1] broadcast operand's 1 valid col) and the blayout
        (leaking col_major from the shape-1 operand into the [128,128] result).
        """
        m = _index_scalar("M")
        # at is the full-shaped [128,128] operand (row_major); bt is a [128,1]
        # column-broadcast operand carrying col_major (as [N,1] tiles infer).
        at = _mk_tile("at", [128, 128], [m, 64], blayout=ir.TileLayout.row_major)
        bt = _mk_tile("bt", [128, 1], [m, 1], blayout=ir.TileLayout.col_major)

        r1 = tile.add(at, bt).type
        r2 = tile.add(bt, at).type
        v1 = _effective_valid(r1)
        v2 = _effective_valid(r2)
        # Row valid is the shared dynamic M on both operands; col valid is at's 64
        # (bt is broadcast in the col dim => exempt, contributes nothing).
        assert v1[0] is m and v2[0] is m
        assert isinstance(v1[1], ir.ConstInt) and v1[1].value == 64
        assert isinstance(v2[1], ir.ConstInt) and v2[1].value == 64
        # Commutativity: identical valid_shape and blayout regardless of arg order.
        assert ir.structural_equal(r1, r2)
        assert _blayout(r1) == _blayout(r2) == ir.TileLayout.row_major
        # The result must NOT narrow to the broadcast operand's 1 valid col.
        assert not (isinstance(v2[1], ir.ConstInt) and v2[1].value == 1)

    def test_fully_valid_broadcast_unchanged(self):
        """Regression guard: a fully-valid [128,1] + [128,128] stays fully valid and
        canonicalizes to no view — the overwhelmingly common case is byte-identical."""
        rr = tile.add(_mk_tile("a", [128, 1]), _mk_tile("b", [128, 128])).type
        assert isinstance(rr, ir.TileType)
        assert _valid_ints(rr) == [128, 128]
        assert rr.tile_view is None

    def test_matching_shape_equal_partial_valid_propagates(self):
        """Two same-shaped operands with the SAME partial valid propagate it."""
        rr = tile.add(_mk_tile("a", [128, 128], [64, 32]), _mk_tile("b", [128, 128], [64, 32])).type
        assert _valid_ints(rr) == [64, 32]

    def test_matching_shape_provably_unequal_static_valid_rejects(self):
        """A provable static valid mismatch (64 vs 32) on a non-broadcast dim is a
        user error, rejected loudly (valid_shape never broadcasts)."""
        m = _index_scalar("M")
        a = _mk_tile("a", [128, 128], [m, 64])
        b = _mk_tile("b", [128, 128], [m, 32])
        with pytest.raises(ValueError, match="disagree on the valid extent along dim 1"):
            tile.add(a, b)

    def test_symbolic_valid_not_provably_equal_accepted(self):
        """Dynamic-valid regression guard: distinct symbolic row extents cannot be
        proved to disagree and must NOT be rejected (dynamic valid_shape works)."""
        m = _index_scalar("M")
        n = _index_scalar("N")
        a = _mk_tile("a", [128, 128], [m, 64])
        b = _mk_tile("b", [128, 128], [n, 64])
        result_type = tile.add(a, b).type  # must not raise
        assert isinstance(result_type, ir.TileType)
        # Col agrees at 64; row is a shared (assumed-equal) symbolic extent.
        valid = _effective_valid(result_type)
        assert isinstance(valid[1], ir.ConstInt) and valid[1].value == 64

    def test_broadcast_operand_valid_zero_is_discarded(self):
        """A broadcast operand (physical shape 1) with a 0 valid extent in its
        broadcast dim imposes NO constraint — valid_shape never broadcasts, so the
        output takes the non-broadcast operand's extent, not 0."""
        m = _index_scalar("M")
        bt = _mk_tile("bt", [128, 1], [m, 0])  # 0 valid in the (broadcast) col dim
        at = _mk_tile("at", [128, 128], [m, 64])
        result_type = tile.add(bt, at).type
        valid = _effective_valid(result_type)
        # The col extent is at's 64 — bt's 0 broadcast extent is discarded, NOT
        # broadcast into an empty result.
        assert isinstance(valid[1], ir.ConstInt) and valid[1].value == 64

    def test_shift_binary_agrees_on_valid(self):
        """Shift (dst[i]=lhs[i]<<rhs[i]) requires lhs and the shift-amount rhs to agree."""
        lhs = _mk_tile("lhs", [128, 128], [40, 64], dtype=DataType.INT32)
        rhs = _mk_tile("rhs", [128, 128], [40, 64], dtype=DataType.INT32)
        assert _valid_ints(tile.shl(lhs, rhs).type) == [40, 64]
        # A provable shift-amount valid mismatch rejects.
        bad = _mk_tile("bad", [128, 128], [40, 32], dtype=DataType.INT32)
        with pytest.raises(ValueError, match="disagree on the valid extent"):
            tile.shl(lhs, bad)

    def test_ternary_agrees_on_value_operands_ignoring_tmp(self):
        """rem (lhs, rhs, tmp): agreement over lhs/rhs; the scratch tmp is ignored."""
        lhs = _mk_tile("lhs", [128, 128], [40, 64])
        rhs = _mk_tile("rhs", [128, 128], [40, 64])
        tmp = _mk_tile("tmp", [128, 128], [8, 8])  # a narrower scratch must NOT constrain
        assert _valid_ints(tile.rem(lhs, rhs, tmp).type) == [40, 64]

    def test_tri_tile_agrees_across_all_three(self):
        """addc (lhs, rhs, rhs2): all three real value tiles must agree."""
        a = _mk_tile("a", [128, 128], [40, 64])
        b = _mk_tile("b", [128, 128], [40, 64])
        c = _mk_tile("c", [128, 128], [40, 64])
        assert _valid_ints(tile.addc(a, b, c).type) == [40, 64]
        bad = _mk_tile("bad", [128, 128], [40, 32])
        with pytest.raises(ValueError, match="disagree on the valid extent"):
            tile.addc(a, b, bad)

    def test_tile_scalar_tile_agrees_on_two_tiles(self):
        """addsc (tile, scalar, tile): the two value tiles agree; the scalar has none."""
        a = _mk_tile("a", [128, 128], [40, 64])
        c = _mk_tile("c", [128, 128], [40, 64])
        assert _valid_ints(tile.addsc(a, 3.0, c).type) == [40, 64]

    def test_sel_agrees_on_value_tiles_ignoring_mask_and_tmp(self):
        """sel (mask, lhs, rhs, tmp): agreement over lhs/rhs only — the ceil_div'd
        mask and the scratch tmp are excluded."""
        mask = _mk_tile("mask", [128, 128], dtype=DataType.UINT8)
        lhs = _mk_tile("lhs", [128, 128], [40, 64])
        rhs = _mk_tile("rhs", [128, 128], [40, 64])
        tmp = _mk_tile("tmp", [1, 32], dtype=DataType.UINT8)
        assert _valid_ints(tile.sel(mask, lhs, rhs, tmp).type) == [40, 64]
        bad = _mk_tile("bad", [128, 128], [40, 32])
        with pytest.raises(ValueError, match="disagree on the valid extent"):
            tile.sel(mask, lhs, bad, tmp)

    def test_sel_scalar_agrees_on_two_tiles(self):
        """sels (lhs, rhs, mode): the two value tiles agree; the scalar mode has none."""
        lhs = _mk_tile("lhs", [128, 128], [40, 64])
        rhs = _mk_tile("rhs", [128, 128], [40, 64])
        assert _valid_ints(tile.sels(lhs, rhs, 0).type) == [40, 64]

    def test_cmp_mask_valid_cols_ceil_div_of_agreed_logical_valid(self):
        """cmp (2 tiles): the packed mask's valid cols = ceil_div(agreed logical valid
        cols, 8); the rhs valid region now participates in the agreement."""
        lhs = _mk_tile("lhs", [128, 128], [40, 64])
        rhs = _mk_tile("rhs", [128, 128], [40, 64])
        mask_type = tile.cmp(lhs, rhs).type
        valid = _valid_ints(mask_type)
        # Rows carried through (40); cols packed: ceil_div(64, 8) = 8.
        assert valid == [40, 8]

    def test_cmp_rejects_rhs_valid_mismatch(self):
        """cmp consults BOTH tiles' logical valid — a provable rhs mismatch rejects."""
        lhs = _mk_tile("lhs", [128, 128], [40, 64])
        rhs = _mk_tile("rhs", [128, 128], [40, 32])
        with pytest.raises(ValueError, match="disagree on the valid extent"):
            tile.cmp(lhs, rhs)

    def test_cmps_mask_valid_cols_ceil_div_of_lhs_valid(self):
        """cmps (tile, scalar): only lhs has a valid region; cols = ceil_div(lhs col, 8)."""
        lhs = _mk_tile("lhs", [128, 128], [40, 64])
        mask_type = tile.cmps(lhs, 1.0).type
        assert _valid_ints(mask_type) == [40, 8]

    def test_part_add_union_is_commutative(self):
        """Partial-combine ops (valid where EITHER source is valid) use the origin-
        anchored union, NOT strict agreement, so operand order does not matter."""
        full = _mk_tile("full", [128, 128])  # fully valid [128,128]
        narrow = _mk_tile("narrow", [128, 128], [64, 64])
        u1 = tile.part_add(full, narrow).type
        u2 = tile.part_add(narrow, full).type
        # Union of [128,128] and [64,64] (nested) = [128,128], both orders.
        assert _valid_ints(u1) == _valid_ints(u2) == [128, 128]


class TestTileUnaryOps:
    """Test suite for tile-level unary operators."""

    def test_tile_log(self):
        """Test tile.log operator - natural logarithm of all elements."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(a, [0, 0], [32, 32])
                tile_c: pl.Tile[[32, 32], pl.FP32] = pl.log(tile_a)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.log" in ir_str

    def test_tile_abs(self):
        """Test tile.abs operator - absolute value of all elements."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(a, [0, 0], [32, 32])
                tile_c: pl.Tile[[32, 32], pl.FP32] = pl.abs(tile_a)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.abs" in ir_str

    def test_tile_relu(self):
        """Test tile.relu operator - ReLU activation function."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(a, [0, 0], [32, 32])
                tile_c: pl.Tile[[32, 32], pl.FP32] = pl.relu(tile_a)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.relu" in ir_str

    def test_tile_exp(self):
        """Test tile.exp operator - exponential of all elements."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(a, [0, 0], [32, 32])
                tile_c: pl.Tile[[32, 32], pl.FP32] = pl.exp(tile_a)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.exp" in ir_str

    def test_tile_sqrt(self):
        """Test tile.sqrt operator - square root of all elements."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(a, [0, 0], [32, 32])
                tile_c: pl.Tile[[32, 32], pl.FP32] = pl.sqrt(tile_a)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.sqrt" in ir_str

    def test_tile_rsqrt_rejects_tmp_shape_mismatch(self):
        """tile.rsqrt rejects a tmp tile whose per-dim shape differs from the input."""
        span = ir.Span.unknown()
        input_type = ir.TileType([16, 64], DataType.FP32)
        tmp_type = ir.TileType([32, 64], DataType.FP32)  # rank matches, dim 0 differs
        input_var = ir.Var("src", input_type, span)
        tmp_var = ir.Var("tmp", tmp_type, span)

        with pytest.raises(ValueError, match="shape mismatch"):
            tile.rsqrt(input_var, tmp_var)

    def test_tile_rsqrt_high_precision(self):
        """tile.rsqrt accepts an optional tmp tile for the high-precision path."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(a, [0, 0], [32, 32])
                tmp: pl.Tile[[32, 32], pl.FP32] = pl.tile.create(
                    [32, 32], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
                )
                tile_c: pl.Tile[[32, 32], pl.FP32] = pl.tile.rsqrt(tile_a, tmp=tmp)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.rsqrt" in ir_str

    def test_tile_neg(self):
        """Test tile.neg operator - negate all elements."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(a, [0, 0], [32, 32])
                tile_c: pl.Tile[[32, 32], pl.FP32] = pl.neg(tile_a)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.neg" in ir_str

    # ------------------------------------------------------------------
    # tile.sin / tile.cos (FP32-only)
    # ------------------------------------------------------------------

    def test_tile_sin_creates_call(self):
        """tile.sin on an FP32 tile produces a Call with FP32 output of the same shape."""
        span = ir.Span.unknown()
        tile_type = ir.TileType([32, 64], DataType.FP32)
        tile_var = ir.Var("x", tile_type, span)

        call = tile.sin(tile_var)

        assert isinstance(call, ir.Call)
        assert call.op.name == "tile.sin"

        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        assert result_type.dtype == DataType.FP32
        assert len(result_type.shape) == 2

    def test_tile_cos_creates_call(self):
        """tile.cos on an FP32 tile produces a Call with FP32 output of the same shape."""
        span = ir.Span.unknown()
        tile_type = ir.TileType([32, 64], DataType.FP32)
        tile_var = ir.Var("x", tile_type, span)

        call = tile.cos(tile_var)

        assert isinstance(call, ir.Call)
        assert call.op.name == "tile.cos"

        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        assert result_type.dtype == DataType.FP32
        assert len(result_type.shape) == 2

    def test_tile_sin_rejects_fp16(self):
        """tile.sin must reject FP16 input with an error mentioning the op name and FP32."""
        span = ir.Span.unknown()
        tile_type = ir.TileType([32, 64], DataType.FP16)
        tile_var = ir.Var("x", tile_type, span)

        with pytest.raises(ValueError, match=r"tile\.sin.*FP32"):
            tile.sin(tile_var)

    def test_tile_cos_rejects_bf16(self):
        """tile.cos must reject BF16 input with an error mentioning the op name and FP32."""
        span = ir.Span.unknown()
        tile_type = ir.TileType([32, 64], DataType.BF16)
        tile_var = ir.Var("x", tile_type, span)

        with pytest.raises(ValueError, match=r"tile\.cos.*FP32"):
            tile.cos(tile_var)

    def test_tile_sin_rejects_int32(self):
        """tile.sin must reject INT32 input with an FP32-mentioning error."""
        span = ir.Span.unknown()
        tile_type = ir.TileType([32, 64], DataType.INT32)
        tile_var = ir.Var("x", tile_type, span)

        with pytest.raises(ValueError, match=r"(?i)FP32"):
            tile.sin(tile_var)

    # ------------------------------------------------------------------
    # Issue #1370: unary tile ops must preserve TileView.valid_shape
    # from their input. Without this, chains like
    #   pl.slice(..., valid_shape=[16, 4]) -> pl.tile.muls -> pl.tile.neg
    # cause codegen to emit dst.validCol=8 against src.validCol=4 and the
    # NPU produces wrong outputs.
    # ------------------------------------------------------------------

    def _make_sliced_tile_with_valid_shape(self):
        """Helper: returns a tile-typed Call whose result has valid_shape=[8, 4]."""
        span = ir.Span.unknown()
        src_type = ir.TileType(
            [ir.ConstInt(8, DataType.INT32, span), ir.ConstInt(16, DataType.INT32, span)],
            DataType.FP32,
        )
        src_var = ir.Var("src", src_type, span)
        return tile.slice(src_var, [8, 16], [0, 0], valid_shape=[8, 4])

    def test_tile_neg_preserves_input_valid_shape(self):
        """tile.neg must propagate the source TileView's valid_shape (issue #1370)."""
        sliced = self._make_sliced_tile_with_valid_shape()
        call = tile.neg(sliced)
        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        assert result_type.tile_view is not None
        valid_shape = result_type.tile_view.valid_shape
        assert len(valid_shape) == 2
        assert isinstance(valid_shape[0], ir.ConstInt) and valid_shape[0].value == 8
        assert isinstance(valid_shape[1], ir.ConstInt) and valid_shape[1].value == 4

    def test_tile_exp_preserves_input_valid_shape(self):
        """tile.exp must propagate the source TileView's valid_shape (issue #1370)."""
        sliced = self._make_sliced_tile_with_valid_shape()
        call = tile.exp(sliced)
        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        assert result_type.tile_view is not None
        valid_shape = result_type.tile_view.valid_shape
        assert isinstance(valid_shape[1], ir.ConstInt) and valid_shape[1].value == 4

    def test_tile_cast_preserves_input_valid_shape(self):
        """tile.cast must propagate the source TileView's valid_shape (issue #1370)."""
        sliced = self._make_sliced_tile_with_valid_shape()
        call = tile.cast(sliced, DataType.FP16)
        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        assert result_type.dtype == DataType.FP16
        assert result_type.tile_view is not None
        valid_shape = result_type.tile_view.valid_shape
        assert isinstance(valid_shape[1], ir.ConstInt) and valid_shape[1].value == 4

    def test_tile_cast_rejects_same_dtype(self):
        """tile.cast must reject same-dtype invocation at construction time.

        Hardware pto.tcvt is for cross-dtype conversion; a same-dtype cast (e.g.
        FP32 -> FP32) can corrupt values rather than acting as an identity copy.
        DeduceTileCastType raises so malformed casts never reach any pass or codegen.
        """
        span = ir.Span.unknown()
        src_type = ir.TileType(
            [ir.ConstInt(8, DataType.INT32, span), ir.ConstInt(16, DataType.INT32, span)],
            DataType.FP32,
        )
        src_var = ir.Var("src", src_type, span)

        with pytest.raises(ValueError, match="same-dtype cast is not a valid operation"):
            tile.cast(src_var, DataType.FP32)

    def test_tile_rsqrt_preserves_input_valid_shape(self):
        """tile.rsqrt must propagate the source TileView's valid_shape (issue #1370)."""
        sliced = self._make_sliced_tile_with_valid_shape()
        call = tile.rsqrt(sliced)
        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        assert result_type.tile_view is not None
        valid_shape = result_type.tile_view.valid_shape
        assert isinstance(valid_shape[1], ir.ConstInt) and valid_shape[1].value == 4

    def test_tile_not_preserves_input_valid_shape(self):
        """tile.not must propagate the source TileView's valid_shape (issue #1370)."""
        span = ir.Span.unknown()
        src_type = ir.TileType(
            [ir.ConstInt(8, DataType.INT32, span), ir.ConstInt(16, DataType.INT32, span)],
            DataType.INT16,
        )
        src_var = ir.Var("src", src_type, span)
        sliced = tile.slice(src_var, [8, 16], [0, 0], valid_shape=[8, 4])
        call = tile.not_(sliced)
        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        assert result_type.tile_view is not None
        valid_shape = result_type.tile_view.valid_shape
        assert isinstance(valid_shape[1], ir.ConstInt) and valid_shape[1].value == 4


class TestTileReductionOps:
    """Test suite for tile-level reduction operators."""

    def test_tile_sum_axis0(self):
        """Test tile.sum operator - sum along axis 0 (column-wise)."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(a, [0, 0], [32, 32])
                tile_c: pl.Tile[[32], pl.FP32] = pl.sum(tile_a, axis=0)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.sum" in ir_str

    def test_tile_sum_axis1(self):
        """Test tile.sum operator - sum along axis 1 (row-wise)."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(a, [0, 0], [32, 32])
                tile_c: pl.Tile[[32], pl.FP32] = pl.sum(tile_a, axis=1)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.sum" in ir_str

    def test_tile_max_axis0(self):
        """Test tile.max operator - max along axis 0 (column-wise)."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(a, [0, 0], [32, 32])
                tile_c: pl.Tile[[32], pl.FP32] = pl.max(tile_a, axis=0)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.max" in ir_str

    def test_tile_max_axis1(self):
        """Test tile.max operator - max along axis 1 (row-wise)."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(a, [0, 0], [32, 32])
                tile_c: pl.Tile[[32], pl.FP32] = pl.max(tile_a, axis=1)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.max" in ir_str

    def test_tile_row_max(self, ascend_backend, default_pass_manager):
        """Test tile.row_max operation."""

        @pl.program
        class RowMaxKernel:
            @pl.function(type=pl.FunctionType.InCore)
            def row_max_kernel(
                self, input: pl.Tensor[[128, 128], pl.FP32], output: pl.Tensor[[128, 1], pl.FP32]
            ) -> pl.Tensor[[128, 1], pl.FP32]:
                tile_in: pl.Tile[[32, 128], pl.FP32] = pl.load(input, [0, 0], [32, 128])
                tmp_tile: pl.Tile[[32, 1], pl.FP32] = pl.tile.create(
                    [32, 1], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
                )
                tile_max: pl.Tile[[32, 1], pl.FP32] = pl.row_max(tile_in, tmp_tile)
                result: pl.Tensor[[128, 1], pl.FP32] = pl.store(tile_max, [0, 0], output)
                return result

        optimized_program = default_pass_manager.run_passes(RowMaxKernel)

        assert optimized_program is not None
        assert "tile.row_max" in str(optimized_program)

    def test_tile_row_sum(self, ascend_backend, default_pass_manager):
        """Test tile.row_sum operation."""

        @pl.program
        class RowSumKernel:
            @pl.function(type=pl.FunctionType.InCore)
            def row_sum_kernel(
                self, input: pl.Tensor[[128, 128], pl.FP32], output: pl.Tensor[[128, 1], pl.FP32]
            ) -> pl.Tensor[[128, 1], pl.FP32]:
                tile_in: pl.Tile[[32, 128], pl.FP32] = pl.load(input, [0, 0], [32, 128])
                tmp_tile: pl.Tile[[32, 1], pl.FP32] = pl.tile.create(
                    [32, 1], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
                )
                tile_sum: pl.Tile[[32, 1], pl.FP32] = pl.row_sum(tile_in, tmp_tile)
                result: pl.Tensor[[128, 1], pl.FP32] = pl.store(tile_sum, [0, 0], output)
                return result

        optimized_program = default_pass_manager.run_passes(RowSumKernel)

        assert optimized_program is not None
        assert "tile.row_sum" in str(optimized_program)

    def test_tile_row_min(self):
        """Test tile.row_min operation."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                input: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 1], pl.FP32],
            ) -> pl.Tensor[[128, 1], pl.FP32]:
                tile_in: pl.Tile[[32, 128], pl.FP32] = pl.load(input, [0, 0], [32, 128])
                tmp_tile: pl.Tile[[32, 128], pl.FP32] = pl.tile.create(
                    [32, 128], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
                )
                tile_row_min: pl.Tile[[32, 1], pl.FP32] = pl.row_min(tile_in, tmp_tile)
                result: pl.Tensor[[128, 1], pl.FP32] = pl.store(tile_row_min, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.row_min" in ir_str

    def test_tile_col_sum(self):
        """Test tile.col_sum operation (2 args: tile + tmp_tile)."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                input: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[1, 128], pl.FP32],
            ) -> pl.Tensor[[1, 128], pl.FP32]:
                tile_in: pl.Tile[[32, 128], pl.FP32] = pl.load(input, [0, 0], [32, 128])
                tmp_tile: pl.Tile[[32, 128], pl.FP32] = pl.tile.create(
                    [32, 128], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
                )
                tile_col_sum: pl.Tile[[1, 128], pl.FP32] = pl.tile.col_sum(tile_in, tmp_tile)
                result: pl.Tensor[[1, 128], pl.FP32] = pl.store(tile_col_sum, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.col_sum" in ir_str

    def test_tile_col_max(self):
        """Test tile.col_max operation (1 arg)."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                input: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[1, 128], pl.FP32],
            ) -> pl.Tensor[[1, 128], pl.FP32]:
                tile_in: pl.Tile[[32, 128], pl.FP32] = pl.load(input, [0, 0], [32, 128])
                tile_col_max: pl.Tile[[1, 128], pl.FP32] = pl.tile.col_max(tile_in)
                result: pl.Tensor[[1, 128], pl.FP32] = pl.store(tile_col_max, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.col_max" in ir_str

    def test_tile_col_min(self):
        """Test tile.col_min operation (1 arg)."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                input: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[1, 128], pl.FP32],
            ) -> pl.Tensor[[1, 128], pl.FP32]:
                tile_in: pl.Tile[[32, 128], pl.FP32] = pl.load(input, [0, 0], [32, 128])
                tile_col_min: pl.Tile[[1, 128], pl.FP32] = pl.tile.col_min(tile_in)
                result: pl.Tensor[[1, 128], pl.FP32] = pl.store(tile_col_min, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.col_min" in ir_str

    def test_tile_row_prod(self):
        """Test tile.row_prod operation (2 args: tile + tmp_tile)."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                input: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 1], pl.FP32],
            ) -> pl.Tensor[[128, 1], pl.FP32]:
                tile_in: pl.Tile[[32, 128], pl.FP32] = pl.load(input, [0, 0], [32, 128])
                tmp_tile: pl.Tile[[32, 128], pl.FP32] = pl.tile.create(
                    [32, 128], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
                )
                tile_row_prod: pl.Tile[[32, 1], pl.FP32] = pl.row_prod(tile_in, tmp_tile)
                result: pl.Tensor[[128, 1], pl.FP32] = pl.store(tile_row_prod, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.row_prod" in ir_str

    def test_tile_col_prod(self):
        """Test tile.col_prod operation (1 arg)."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                input: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[1, 128], pl.FP32],
            ) -> pl.Tensor[[1, 128], pl.FP32]:
                tile_in: pl.Tile[[32, 128], pl.FP32] = pl.load(input, [0, 0], [32, 128])
                tile_col_prod: pl.Tile[[1, 128], pl.FP32] = pl.tile.col_prod(tile_in)
                result: pl.Tensor[[1, 128], pl.FP32] = pl.store(tile_col_prod, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.col_prod" in ir_str

    def test_tile_row_argmax(self):
        """Test tile.row_argmax (2 args, int32 index output)."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                input: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 1], pl.INT32],
            ) -> pl.Tensor[[128, 1], pl.INT32]:
                tile_in: pl.Tile[[32, 128], pl.FP32] = pl.load(input, [0, 0], [32, 128])
                tmp_tile: pl.Tile[[32, 128], pl.FP32] = pl.tile.create(
                    [32, 128], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
                )
                tile_argmax: pl.Tile[[32, 1], pl.INT32] = pl.row_argmax(tile_in, tmp_tile)
                result: pl.Tensor[[128, 1], pl.INT32] = pl.store(tile_argmax, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.row_argmax" in ir_str

    def test_tile_row_argmin(self):
        """Test tile.row_argmin (2 args, int32 index output)."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                input: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 1], pl.INT32],
            ) -> pl.Tensor[[128, 1], pl.INT32]:
                tile_in: pl.Tile[[32, 128], pl.FP32] = pl.load(input, [0, 0], [32, 128])
                tmp_tile: pl.Tile[[32, 128], pl.FP32] = pl.tile.create(
                    [32, 128], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
                )
                tile_argmin: pl.Tile[[32, 1], pl.INT32] = pl.row_argmin(tile_in, tmp_tile)
                result: pl.Tensor[[128, 1], pl.INT32] = pl.store(tile_argmin, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.row_argmin" in ir_str

    def test_tile_col_argmax(self):
        """Test tile.col_argmax (2 args incl. tmp, int32 index output)."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                input: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[1, 128], pl.INT32],
            ) -> pl.Tensor[[1, 128], pl.INT32]:
                tile_in: pl.Tile[[32, 128], pl.FP32] = pl.load(input, [0, 0], [32, 128])
                tmp_tile: pl.Tile[[32, 128], pl.FP32] = pl.tile.create(
                    [32, 128], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
                )
                tile_argmax: pl.Tile[[1, 128], pl.INT32] = pl.col_argmax(tile_in, tmp_tile)
                result: pl.Tensor[[1, 128], pl.INT32] = pl.store(tile_argmax, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.col_argmax" in ir_str

    def test_tile_col_argmin(self):
        """Test tile.col_argmin (2 args incl. tmp, int32 index output)."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                input: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[1, 128], pl.INT32],
            ) -> pl.Tensor[[1, 128], pl.INT32]:
                tile_in: pl.Tile[[32, 128], pl.FP32] = pl.load(input, [0, 0], [32, 128])
                tmp_tile: pl.Tile[[32, 128], pl.FP32] = pl.tile.create(
                    [32, 128], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
                )
                tile_argmin: pl.Tile[[1, 128], pl.INT32] = pl.col_argmin(tile_in, tmp_tile)
                result: pl.Tensor[[1, 128], pl.INT32] = pl.store(tile_argmin, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.col_argmin" in ir_str

    def test_tile_min_axis0(self):
        """Test tile.min operator - min along axis 0 (column-wise)."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(a, [0, 0], [32, 32])
                tile_c: pl.Tile[[32], pl.FP32] = pl.min(tile_a, axis=0)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.min" in ir_str

    def test_tile_min_axis1(self):
        """Test tile.min operator - min along axis 1 (row-wise)."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(a, [0, 0], [32, 32])
                tile_c: pl.Tile[[32], pl.FP32] = pl.min(tile_a, axis=1)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.min" in ir_str

    # ------------------------------------------------------------------
    # Issue #1401: reduction tile ops must inherit TileView.valid_shape
    # from their input along non-reduced dims. Without this, chains like
    #   pl.slice(..., valid_shape=[4, 32]) -> tile.cast -> tile.mul -> tile.row_sum
    # cause codegen to emit trowsum with valid_row = physical_rows (e.g. 8)
    # against a tcvt/tmul that only wrote `valid_row = 4` rows, picking up
    # uninitialised LB residue on the unwritten rows.
    # ------------------------------------------------------------------

    def _make_sliced_tile_with_valid_shape(self, valid_rows=4, valid_cols=32):
        """Helper: returns a tile-typed Call with valid_shape=[valid_rows, valid_cols]."""
        span = ir.Span.unknown()
        src_type = ir.TileType(
            [ir.ConstInt(8, DataType.INT32, span), ir.ConstInt(32, DataType.INT32, span)],
            DataType.FP32,
        )
        src_var = ir.Var("src", src_type, span)
        return tile.slice(src_var, [8, 32], [0, 0], valid_shape=[valid_rows, valid_cols])

    def test_tile_row_sum_inherits_input_valid_shape(self):
        """tile.row_sum output valid_shape must mirror input on the kept dim (issue #1401)."""
        sliced = self._make_sliced_tile_with_valid_shape(valid_rows=4, valid_cols=32)
        span = ir.Span.unknown()
        tmp_type = ir.TileType(
            [ir.ConstInt(8, DataType.INT32, span), ir.ConstInt(1, DataType.INT32, span)],
            DataType.FP32,
        )
        tmp_var = ir.Var("tmp", tmp_type, span)

        call = tile.row_sum(sliced, tmp_var)
        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        assert result_type.tile_view is not None
        valid_shape = result_type.tile_view.valid_shape
        # Output: [rows=4 (kept, inherited from input valid_shape), 1 (reduced)]
        assert len(valid_shape) == 2
        assert isinstance(valid_shape[0], ir.ConstInt) and valid_shape[0].value == 4
        assert isinstance(valid_shape[1], ir.ConstInt) and valid_shape[1].value == 1

    def test_tile_row_max_inherits_input_valid_shape(self):
        """tile.row_max must inherit valid_shape from input (issue #1401)."""
        sliced = self._make_sliced_tile_with_valid_shape(valid_rows=4, valid_cols=32)
        span = ir.Span.unknown()
        tmp_type = ir.TileType(
            [ir.ConstInt(8, DataType.INT32, span), ir.ConstInt(1, DataType.INT32, span)],
            DataType.FP32,
        )
        tmp_var = ir.Var("tmp", tmp_type, span)
        call = tile.row_max(sliced, tmp_var)
        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        assert result_type.tile_view is not None
        valid_shape = result_type.tile_view.valid_shape
        assert isinstance(valid_shape[0], ir.ConstInt) and valid_shape[0].value == 4
        assert isinstance(valid_shape[1], ir.ConstInt) and valid_shape[1].value == 1

    def test_tile_col_sum_inherits_input_valid_shape(self):
        """tile.col_sum output valid_shape must mirror input on the kept dim (issue #1401)."""
        sliced = self._make_sliced_tile_with_valid_shape(valid_rows=4, valid_cols=16)
        # col_sum takes 1 arg
        call = tile.col_sum(sliced)
        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        assert result_type.tile_view is not None
        valid_shape = result_type.tile_view.valid_shape
        # Output: [1 (reduced), cols=16 (kept, inherited from input valid_shape)]
        assert len(valid_shape) == 2
        assert isinstance(valid_shape[0], ir.ConstInt) and valid_shape[0].value == 1
        assert isinstance(valid_shape[1], ir.ConstInt) and valid_shape[1].value == 16

    def test_tile_col_max_inherits_input_valid_shape(self):
        """tile.col_max must inherit valid_shape from input (issue #1401)."""
        sliced = self._make_sliced_tile_with_valid_shape(valid_rows=4, valid_cols=16)
        call = tile.col_max(sliced)
        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        assert result_type.tile_view is not None
        valid_shape = result_type.tile_view.valid_shape
        assert isinstance(valid_shape[0], ir.ConstInt) and valid_shape[0].value == 1
        assert isinstance(valid_shape[1], ir.ConstInt) and valid_shape[1].value == 16

    def test_tile_sum_axis_keepdim_inherits_input_valid_shape(self):
        """tile.sum(axis=1, keepdim=True) must inherit valid_shape (issue #1401)."""
        sliced = self._make_sliced_tile_with_valid_shape(valid_rows=4, valid_cols=32)
        call = tile.sum(sliced, axis=1, keepdim=True)
        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        assert result_type.tile_view is not None
        valid_shape = result_type.tile_view.valid_shape
        # Output: [rows=4 (kept), 1 (reduced with keepdim)]
        assert len(valid_shape) == 2
        assert isinstance(valid_shape[0], ir.ConstInt) and valid_shape[0].value == 4
        assert isinstance(valid_shape[1], ir.ConstInt) and valid_shape[1].value == 1

    def test_tile_sum_axis_no_keepdim_inherits_input_valid_shape(self):
        """tile.sum(axis=1, keepdim=False) drops the reduced dim; kept dim inherits valid_shape."""
        sliced = self._make_sliced_tile_with_valid_shape(valid_rows=4, valid_cols=32)
        call = tile.sum(sliced, axis=1, keepdim=False)
        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        assert result_type.tile_view is not None
        valid_shape = result_type.tile_view.valid_shape
        # Output: [rows=4 (kept)] — reduced dim is dropped entirely
        assert len(valid_shape) == 1
        assert isinstance(valid_shape[0], ir.ConstInt) and valid_shape[0].value == 4


class TestTileBroadcastOps:
    """Test suite for tile-level broadcast operators."""

    def test_tile_col_expand(self):
        """Test tile.col_expand operator - expand column vector to target shape."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                target: pl.Tensor[[128, 128], pl.FP32],
                col: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_target: pl.Tile[[32, 32], pl.FP32] = pl.load(target, [0, 0], [32, 32])
                tile_col: pl.Tile[[1, 32], pl.FP32] = pl.load(col, [0, 0], [1, 32])
                tile_c: pl.Tile[[32, 32], pl.FP32] = pl.col_expand(tile_target, tile_col)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.col_expand" in ir_str

    def test_tile_col_expand_mul(self):
        """Test tile.col_expand_mul operator - expand column and multiply with tile."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                col: pl.Tensor[[128, 128], pl.FP32],
                tile: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_col: pl.Tile[[1, 32], pl.FP32] = pl.load(col, [0, 0], [1, 32])
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(tile, [0, 0], [32, 32])
                tile_c: pl.Tile[[32, 32], pl.FP32] = pl.col_expand_mul(tile_a, tile_col)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.col_expand_mul" in ir_str

    def test_tile_col_expand_div(self):
        """Test tile.col_expand_div operator - expand column and divide tile."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                col: pl.Tensor[[128, 128], pl.FP32],
                tile: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_col: pl.Tile[[1, 32], pl.FP32] = pl.load(col, [0, 0], [1, 32])
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(tile, [0, 0], [32, 32])
                tile_c: pl.Tile[[32, 32], pl.FP32] = pl.col_expand_div(tile_a, tile_col)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.col_expand_div" in ir_str

    def test_tile_col_expand_sub(self):
        """Test tile.col_expand_sub operator - expand column and subtract from tile."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                col: pl.Tensor[[128, 128], pl.FP32],
                tile: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_col: pl.Tile[[1, 32], pl.FP32] = pl.load(col, [0, 0], [1, 32])
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(tile, [0, 0], [32, 32])
                tile_c: pl.Tile[[32, 32], pl.FP32] = pl.col_expand_sub(tile_a, tile_col)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.col_expand_sub" in ir_str

    def test_tile_col_expand_add(self):
        """Test tile.col_expand_add operator - expand column and add to tile."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                col: pl.Tensor[[128, 128], pl.FP32],
                tile: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_col: pl.Tile[[1, 32], pl.FP32] = pl.load(col, [0, 0], [1, 32])
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(tile, [0, 0], [32, 32])
                tile_c: pl.Tile[[32, 32], pl.FP32] = pl.col_expand_add(tile_a, tile_col)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.col_expand_add" in ir_str

    def test_tile_row_expand_add(self):
        """Test tile.row_expand_add operator - expand row and add to tile."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                tile: pl.Tensor[[128, 128], pl.FP32],
                row: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(tile, [0, 0], [32, 32])
                tile_row: pl.Tile[[32, 1], pl.FP32] = pl.load(row, [0, 0], [32, 1])
                tile_c: pl.Tile[[32, 32], pl.FP32] = pl.row_expand_add(tile_a, tile_row)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.row_expand_add" in ir_str

    def test_tile_row_expand_sub(self):
        """Test tile.row_expand_sub operator - subtract row vector from each tile row."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                tile: pl.Tensor[[128, 128], pl.FP32],
                row: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(tile, [0, 0], [32, 32])
                tile_row: pl.Tile[[32, 1], pl.FP32] = pl.load(row, [0, 0], [32, 1])
                tile_c: pl.Tile[[32, 32], pl.FP32] = pl.row_expand_sub(tile_a, tile_row)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.row_expand_sub" in ir_str

    def test_tile_row_expand_div(self):
        """Test tile.row_expand_div operator - divide each tile row by row vector."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                tile: pl.Tensor[[128, 128], pl.FP32],
                row: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(tile, [0, 0], [32, 32])
                tile_row: pl.Tile[[32, 1], pl.FP32] = pl.load(row, [0, 0], [32, 1])
                tile_c: pl.Tile[[32, 32], pl.FP32] = pl.row_expand_div(tile_a, tile_row)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.row_expand_div" in ir_str

    def test_tile_row_expand_mul(self):
        """Test tile.row_expand_mul operator - multiply each tile row by row vector."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                tile: pl.Tensor[[128, 128], pl.FP32],
                row: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(tile, [0, 0], [32, 32])
                tile_row: pl.Tile[[32, 1], pl.FP32] = pl.load(row, [0, 0], [32, 1])
                tile_c: pl.Tile[[32, 32], pl.FP32] = pl.row_expand_mul(tile_a, tile_row)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.row_expand_mul" in ir_str

    def test_tile_row_expand_max(self):
        """Test tile.row_expand_max operator - max of each tile row and row vector."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                tile: pl.Tensor[[128, 128], pl.FP32],
                row: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(tile, [0, 0], [32, 32])
                tile_row: pl.Tile[[32, 1], pl.FP32] = pl.load(row, [0, 0], [32, 1])
                tile_c: pl.Tile[[32, 32], pl.FP32] = pl.row_expand_max(tile_a, tile_row)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.row_expand_max" in ir_str

    def test_tile_row_expand_min(self):
        """Test tile.row_expand_min operator - min of each tile row and row vector."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                tile: pl.Tensor[[128, 128], pl.FP32],
                row: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(tile, [0, 0], [32, 32])
                tile_row: pl.Tile[[32, 1], pl.FP32] = pl.load(row, [0, 0], [32, 1])
                tile_c: pl.Tile[[32, 32], pl.FP32] = pl.row_expand_min(tile_a, tile_row)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.row_expand_min" in ir_str

    def test_tile_row_expand_expdif(self):
        """Test tile.row_expand_expdif operator - exp(tile - row vector) per row."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                tile: pl.Tensor[[128, 128], pl.FP32],
                row: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(tile, [0, 0], [32, 32])
                tile_row: pl.Tile[[32, 1], pl.FP32] = pl.load(row, [0, 0], [32, 1])
                tile_c: pl.Tile[[32, 32], pl.FP32] = pl.row_expand_expdif(tile_a, tile_row)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.row_expand_expdif" in ir_str

    def test_tile_col_expand_max(self):
        """Test tile.col_expand_max operator - max of each tile column and col vector."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                tile: pl.Tensor[[128, 128], pl.FP32],
                col: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(tile, [0, 0], [32, 32])
                tile_col: pl.Tile[[1, 32], pl.FP32] = pl.load(col, [0, 0], [1, 32])
                tile_c: pl.Tile[[32, 32], pl.FP32] = pl.col_expand_max(tile_a, tile_col)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.col_expand_max" in ir_str

    def test_tile_col_expand_min(self):
        """Test tile.col_expand_min operator - min of each tile column and col vector."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                tile: pl.Tensor[[128, 128], pl.FP32],
                col: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(tile, [0, 0], [32, 32])
                tile_col: pl.Tile[[1, 32], pl.FP32] = pl.load(col, [0, 0], [1, 32])
                tile_c: pl.Tile[[32, 32], pl.FP32] = pl.col_expand_min(tile_a, tile_col)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.col_expand_min" in ir_str

    def test_tile_col_expand_expdif(self):
        """Test tile.col_expand_expdif operator - exp(tile - col vector) per column."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                tile: pl.Tensor[[128, 128], pl.FP32],
                col: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(tile, [0, 0], [32, 32])
                tile_col: pl.Tile[[1, 32], pl.FP32] = pl.load(col, [0, 0], [1, 32])
                tile_c: pl.Tile[[32, 32], pl.FP32] = pl.col_expand_expdif(tile_a, tile_col)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.col_expand_expdif" in ir_str

    def test_tile_row_expand(self):
        """Test tile.row_expand operator - expand row vector to target tile shape."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                tile: pl.Tensor[[128, 128], pl.FP32],
                row: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(tile, [0, 0], [32, 32])
                tile_row: pl.Tile[[32, 1], pl.FP32] = pl.load(row, [0, 0], [32, 1])
                tile_c: pl.Tile[[32, 32], pl.FP32] = pl.row_expand(tile_a, tile_row)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.row_expand" in ir_str

    def test_tile_expands(self):
        """Test tile.expands operator - expand scalar to tile shape."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(a, [0, 0], [32, 32])
                tile_c: pl.Tile[[32, 32], pl.FP32] = pl.expands(tile_a, 1.0)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.expands" in ir_str

    # ------------------------------------------------------------------
    # Issue #1450: broadcast tile ops must preserve TileView.valid_shape
    # from their main tile input. Without this, chains like
    #   pl.slice(..., valid_shape=[16, 4]) -> pl.row_expand_div(...) -> pl.slice(...)
    # cause the downstream subview verifier to reject the slice with
    # "'pto.subview' op expects result valid_shape[0] to match
    # inferred/explicit valid_row" because row_expand* clobbered the
    # dynamic valid_shape with the static declared shape.
    # ------------------------------------------------------------------

    def _make_sliced_tile_with_valid_shape(self):
        """Helper: returns a tile-typed Call whose result has valid_shape=[8, 4]."""
        span = ir.Span.unknown()
        src_type = ir.TileType(
            [ir.ConstInt(8, DataType.INT32, span), ir.ConstInt(16, DataType.INT32, span)],
            DataType.FP32,
        )
        src_var = ir.Var("src", src_type, span)
        return tile.slice(src_var, [8, 16], [0, 0], valid_shape=[8, 4])

    def _make_row_vec_with_valid_shape(self):
        """Helper: returns a tile-typed Call shaped [8, 1] for row-expand inputs."""
        span = ir.Span.unknown()
        src_type = ir.TileType(
            [ir.ConstInt(8, DataType.INT32, span), ir.ConstInt(1, DataType.INT32, span)],
            DataType.FP32,
        )
        src_var = ir.Var("row_src", src_type, span)
        return tile.slice(src_var, [8, 1], [0, 0], valid_shape=[8, 1])

    def _make_col_vec_with_valid_shape(self):
        """Helper: returns a tile-typed Call shaped [1, 16] for col-expand inputs."""
        span = ir.Span.unknown()
        src_type = ir.TileType(
            [ir.ConstInt(1, DataType.INT32, span), ir.ConstInt(16, DataType.INT32, span)],
            DataType.FP32,
        )
        src_var = ir.Var("col_src", src_type, span)
        return tile.slice(src_var, [1, 16], [0, 0], valid_shape=[1, 4])

    def test_tile_row_expand_div_preserves_input_valid_shape(self):
        """tile.row_expand_div must propagate the main tile's valid_shape (issue #1450)."""
        main = self._make_sliced_tile_with_valid_shape()
        row_vec = self._make_row_vec_with_valid_shape()
        call = tile.row_expand_div(main, row_vec)
        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        assert result_type.tile_view is not None
        valid_shape = result_type.tile_view.valid_shape
        assert len(valid_shape) == 2
        assert isinstance(valid_shape[0], ir.ConstInt) and valid_shape[0].value == 8
        assert isinstance(valid_shape[1], ir.ConstInt) and valid_shape[1].value == 4

    def test_tile_row_expand_mul_preserves_input_valid_shape(self):
        """tile.row_expand_mul must propagate the main tile's valid_shape (issue #1450)."""
        main = self._make_sliced_tile_with_valid_shape()
        row_vec = self._make_row_vec_with_valid_shape()
        call = tile.row_expand_mul(main, row_vec)
        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        assert result_type.tile_view is not None
        valid_shape = result_type.tile_view.valid_shape
        assert isinstance(valid_shape[0], ir.ConstInt) and valid_shape[0].value == 8
        assert isinstance(valid_shape[1], ir.ConstInt) and valid_shape[1].value == 4

    def test_tile_row_expand_sub_preserves_input_valid_shape(self):
        """tile.row_expand_sub must propagate the main tile's valid_shape (issue #1450)."""
        main = self._make_sliced_tile_with_valid_shape()
        row_vec = self._make_row_vec_with_valid_shape()
        call = tile.row_expand_sub(main, row_vec)
        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        assert result_type.tile_view is not None
        valid_shape = result_type.tile_view.valid_shape
        assert isinstance(valid_shape[0], ir.ConstInt) and valid_shape[0].value == 8
        assert isinstance(valid_shape[1], ir.ConstInt) and valid_shape[1].value == 4

    def test_tile_row_expand_add_preserves_input_valid_shape(self):
        """tile.row_expand_add must propagate the main tile's valid_shape (issue #1450)."""
        main = self._make_sliced_tile_with_valid_shape()
        row_vec = self._make_row_vec_with_valid_shape()
        call = tile.row_expand_add(main, row_vec)
        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        assert result_type.tile_view is not None
        valid_shape = result_type.tile_view.valid_shape
        assert isinstance(valid_shape[0], ir.ConstInt) and valid_shape[0].value == 8
        assert isinstance(valid_shape[1], ir.ConstInt) and valid_shape[1].value == 4

    def test_tile_row_expand_preserves_input_valid_shape(self):
        """tile.row_expand must propagate the main tile's valid_shape (issue #1450)."""
        main = self._make_sliced_tile_with_valid_shape()
        row_vec = self._make_row_vec_with_valid_shape()
        call = tile.row_expand(main, row_vec)
        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        assert result_type.tile_view is not None
        valid_shape = result_type.tile_view.valid_shape
        assert isinstance(valid_shape[0], ir.ConstInt) and valid_shape[0].value == 8
        assert isinstance(valid_shape[1], ir.ConstInt) and valid_shape[1].value == 4

    def test_tile_col_expand_mul_preserves_input_valid_shape(self):
        """tile.col_expand_mul must propagate the target tile's valid_shape (issue #1450)."""
        main = self._make_sliced_tile_with_valid_shape()
        col_vec = self._make_col_vec_with_valid_shape()
        call = tile.col_expand_mul(main, col_vec)
        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        assert result_type.tile_view is not None
        valid_shape = result_type.tile_view.valid_shape
        assert isinstance(valid_shape[0], ir.ConstInt) and valid_shape[0].value == 8
        assert isinstance(valid_shape[1], ir.ConstInt) and valid_shape[1].value == 4

    def test_tile_col_expand_div_preserves_input_valid_shape(self):
        """tile.col_expand_div must propagate the target tile's valid_shape (issue #1450)."""
        main = self._make_sliced_tile_with_valid_shape()
        col_vec = self._make_col_vec_with_valid_shape()
        call = tile.col_expand_div(main, col_vec)
        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        assert result_type.tile_view is not None
        valid_shape = result_type.tile_view.valid_shape
        assert isinstance(valid_shape[0], ir.ConstInt) and valid_shape[0].value == 8
        assert isinstance(valid_shape[1], ir.ConstInt) and valid_shape[1].value == 4

    def test_tile_col_expand_sub_preserves_input_valid_shape(self):
        """tile.col_expand_sub must propagate the target tile's valid_shape (issue #1450)."""
        main = self._make_sliced_tile_with_valid_shape()
        col_vec = self._make_col_vec_with_valid_shape()
        call = tile.col_expand_sub(main, col_vec)
        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        assert result_type.tile_view is not None
        valid_shape = result_type.tile_view.valid_shape
        assert isinstance(valid_shape[0], ir.ConstInt) and valid_shape[0].value == 8
        assert isinstance(valid_shape[1], ir.ConstInt) and valid_shape[1].value == 4

    def test_tile_col_expand_add_preserves_input_valid_shape(self):
        """tile.col_expand_add must propagate the target tile's valid_shape (issue #1450)."""
        main = self._make_sliced_tile_with_valid_shape()
        col_vec = self._make_col_vec_with_valid_shape()
        call = tile.col_expand_add(main, col_vec)
        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        assert result_type.tile_view is not None
        valid_shape = result_type.tile_view.valid_shape
        assert isinstance(valid_shape[0], ir.ConstInt) and valid_shape[0].value == 8
        assert isinstance(valid_shape[1], ir.ConstInt) and valid_shape[1].value == 4

    def test_tile_col_expand_preserves_input_valid_shape(self):
        """tile.col_expand must propagate the target tile's valid_shape (issue #1450)."""
        main = self._make_sliced_tile_with_valid_shape()
        col_vec = self._make_col_vec_with_valid_shape()
        call = tile.col_expand(main, col_vec)
        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        assert result_type.tile_view is not None
        valid_shape = result_type.tile_view.valid_shape
        assert isinstance(valid_shape[0], ir.ConstInt) and valid_shape[0].value == 8
        assert isinstance(valid_shape[1], ir.ConstInt) and valid_shape[1].value == 4

    def test_tile_expands_preserves_input_valid_shape(self):
        """tile.expands must propagate the target tile's valid_shape (issue #1450)."""
        main = self._make_sliced_tile_with_valid_shape()
        call = tile.expands(main, 1.0)
        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        assert result_type.tile_view is not None
        valid_shape = result_type.tile_view.valid_shape
        assert isinstance(valid_shape[0], ir.ConstInt) and valid_shape[0].value == 8
        assert isinstance(valid_shape[1], ir.ConstInt) and valid_shape[1].value == 4


class TestTileMatMulOps:
    """Test suite for tile-level matrix multiplication operators."""

    def test_tile_matmul(self):
        """Test tile.matmul operator - matrix multiplication."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 64], pl.FP32],
                b: pl.Tensor[[64, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[32, 16], pl.FP32] = pl.load(a, [0, 0], [32, 16])
                tile_b: pl.Tile[[16, 32], pl.FP32] = pl.load(b, [0, 0], [16, 32])
                tile_c: pl.Tile[[32, 32], pl.FP32] = pl.matmul(tile_a, tile_b)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.matmul" in ir_str

    def test_tile_matmul_acc(self):
        """Test tile.matmul_acc operator - matrix multiplication with accumulation (TMATMUL_ACC).

        Computes: acc_out = acc_in + lhs @ rhs
        """

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                acc_in: pl.Tensor[[128, 128], pl.FP32],
                a: pl.Tensor[[128, 64], pl.FP32],
                b: pl.Tensor[[64, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_acc: pl.Tile[[32, 32], pl.FP32] = pl.load(acc_in, [0, 0], [32, 32])
                tile_a: pl.Tile[[32, 16], pl.FP32] = pl.load(a, [0, 0], [32, 16])
                tile_b: pl.Tile[[16, 32], pl.FP32] = pl.load(b, [0, 0], [16, 32])
                tile_c: pl.Tile[[32, 32], pl.FP32] = pl.matmul_acc(tile_acc, tile_a, tile_b)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.matmul_acc" in ir_str

    def test_tile_matmul_bias(self):
        """Test tile.matmul_bias operator - matrix multiplication with bias add."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 64], pl.FP32],
                b: pl.Tensor[[64, 128], pl.FP32],
                bias: pl.Tensor[[1, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[32, 16], pl.FP32] = pl.load(a, [0, 0], [32, 16])
                tile_b: pl.Tile[[16, 32], pl.FP32] = pl.load(b, [0, 0], [16, 32])
                tile_bias: pl.Tile[[1, 32], pl.FP32] = pl.load(bias, [0, 0], [1, 32])
                tile_c: pl.Tile[[32, 32], pl.FP32] = pl.matmul_bias(tile_a, tile_b, tile_bias)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.matmul_bias" in ir_str

    def test_tile_gemv(self):
        """Test tile.gemv operator - general matrix-vector multiplication."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[1, 64], pl.FP32],
                b: pl.Tensor[[64, 128], pl.FP32],
                output: pl.Tensor[[1, 128], pl.FP32],
            ) -> pl.Tensor[[1, 128], pl.FP32]:
                tile_a: pl.Tile[[1, 16], pl.FP32] = pl.load(a, [0, 0], [1, 16])
                tile_b: pl.Tile[[16, 32], pl.FP32] = pl.load(b, [0, 0], [16, 32])
                tile_c: pl.Tile[[1, 32], pl.FP32] = pl.gemv(tile_a, tile_b)
                result: pl.Tensor[[1, 128], pl.FP32] = pl.store(tile_c, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.gemv" in ir_str

    def test_tile_gemv_acc(self):
        """Test tile.gemv_acc operator - GEMV with accumulation."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                acc_in: pl.Tensor[[1, 128], pl.FP32],
                a: pl.Tensor[[1, 64], pl.FP32],
                b: pl.Tensor[[64, 128], pl.FP32],
                output: pl.Tensor[[1, 128], pl.FP32],
            ) -> pl.Tensor[[1, 128], pl.FP32]:
                tile_acc: pl.Tile[[1, 32], pl.FP32] = pl.load(acc_in, [0, 0], [1, 32])
                tile_a: pl.Tile[[1, 16], pl.FP32] = pl.load(a, [0, 0], [1, 16])
                tile_b: pl.Tile[[16, 32], pl.FP32] = pl.load(b, [0, 0], [16, 32])
                tile_c: pl.Tile[[1, 32], pl.FP32] = pl.gemv_acc(tile_acc, tile_a, tile_b)
                result: pl.Tensor[[1, 128], pl.FP32] = pl.store(tile_c, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.gemv_acc" in ir_str

    def test_tile_gemv_bias(self):
        """Test tile.gemv_bias operator - GEMV with bias add."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[1, 64], pl.FP32],
                b: pl.Tensor[[64, 128], pl.FP32],
                bias: pl.Tensor[[1, 128], pl.FP32],
                output: pl.Tensor[[1, 128], pl.FP32],
            ) -> pl.Tensor[[1, 128], pl.FP32]:
                tile_a: pl.Tile[[1, 16], pl.FP32] = pl.load(a, [0, 0], [1, 16])
                tile_b: pl.Tile[[16, 32], pl.FP32] = pl.load(b, [0, 0], [16, 32])
                tile_bias: pl.Tile[[1, 32], pl.FP32] = pl.load(bias, [0, 0], [1, 32])
                tile_c: pl.Tile[[1, 32], pl.FP32] = pl.gemv_bias(tile_a, tile_b, tile_bias)
                result: pl.Tensor[[1, 128], pl.FP32] = pl.store(tile_c, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.gemv_bias" in ir_str

    # ------------------------------------------------------------------
    # valid_shape propagation: matmul must propagate the
    # M / N valid extents and validate the K contraction. The tile-level ops
    # have NO transpose flag (pl.matmul's a_trans / b_trans apply only on the
    # tensor path and are physically applied before lowering to tile.matmul),
    # so M is always lhs axis 0 and N is always rhs axis 1 here.
    # ------------------------------------------------------------------

    def test_matmul_propagates_m_valid_keeps_dynamic(self):
        """out valid = [valid(lhs)[M], valid(rhs)[N]] — never the full physical shape."""
        m = _index_scalar("M")
        lhs = _mk_tile("lhs", [128, 128], [m, 64])  # valid [M(dyn), K=64]
        rhs = _mk_tile("rhs", [128, 128], [64, 128])  # valid [K=64, N=128]
        result_type = tile.matmul(lhs, rhs).type
        assert isinstance(result_type, ir.TileType)
        assert result_type.tile_view is not None
        valid_shape = result_type.tile_view.valid_shape
        assert len(valid_shape) == 2
        # M valid is the dynamic lhs row extent, carried through unchanged.
        assert valid_shape[0] is m
        # N valid comes from rhs axis 1.
        assert isinstance(valid_shape[1], ir.ConstInt) and valid_shape[1].value == 128
        # The M axis must NOT be widened to the full physical 128.
        assert not (isinstance(valid_shape[0], ir.ConstInt) and valid_shape[0].value == 128)

    def test_matmul_fully_valid_operands_canonicalize(self):
        """Fully-valid operands => fully-valid output; the equal-to-shape view canonicalizes."""
        lhs = _mk_tile("lhs", [128, 128])
        rhs = _mk_tile("rhs", [128, 128])
        result_type = tile.matmul(lhs, rhs).type
        assert isinstance(result_type, ir.TileType)
        # Effective valid == physical shape (nothing narrowed).
        assert _valid_ints(result_type) == [128, 128]
        # Under D2 an equal-to-shape valid_shape collapses: the Acc-stamped view is
        # the implicit Acc view, so the whole view canonicalizes to None.
        assert result_type.tile_view is None

    def test_matmul_rejects_static_valid_k_mismatch(self):
        """A provable valid-K disagreement (64 vs 32) is a user error, rejected loudly."""
        lhs = _mk_tile("lhs", [128, 128], [128, 64])  # valid K = 64
        rhs = _mk_tile("rhs", [128, 128], [32, 128])  # valid K = 32
        with pytest.raises(ValueError, match="disagree on the valid contraction length K"):
            tile.matmul(lhs, rhs)

    def test_matmul_accepts_symbolic_k_structurally_equal(self):
        """The same symbolic K on both operands cannot be a mismatch — accept it."""
        k = _index_scalar("K")
        lhs = _mk_tile("lhs", [128, 128], [128, k])  # valid K = K
        rhs = _mk_tile("rhs", [128, 128], [k, 128])  # valid K = K
        result_type = tile.matmul(lhs, rhs).type  # must not raise
        assert isinstance(result_type, ir.TileType)
        assert _valid_ints(result_type) == [128, 128]

    def test_matmul_accepts_symbolic_k_not_provably_equal(self):
        """Dynamic-K regression guard: distinct symbolic K extents must NOT be rejected."""
        k1 = _index_scalar("K1")
        k2 = _index_scalar("K2")
        lhs = _mk_tile("lhs", [128, 128], [128, k1])
        rhs = _mk_tile("rhs", [128, 128], [k2, 128])
        # Cannot prove k1 != k2, so this stays legal (dynamic K is supported).
        result_type = tile.matmul(lhs, rhs).type  # must not raise
        assert isinstance(result_type, ir.TileType)

    def test_gemv_propagates_m_n_valid_axes(self):
        """gemv [1,K] @ [K,N]: M is lhs axis 0 (=1), N is rhs axis 1."""
        lhs = _mk_tile("lhs", [1, 128], [1, 64])  # valid [M=1, K=64]
        rhs = _mk_tile("rhs", [128, 128], [64, 96])  # valid [K=64, N=96]
        result_type = tile.gemv(lhs, rhs).type
        assert _valid_ints(result_type) == [1, 96]

    def test_matmul_bias_propagates_m_n_valid_axes(self):
        """matmul_bias out valid = matmul valid [M, N]; bias does not change the region."""
        lhs = _mk_tile("lhs", [128, 128], [40, 64])  # valid [M=40, K=64]
        rhs = _mk_tile("rhs", [128, 128], [64, 96])  # valid [K=64, N=96]
        bias = _mk_tile("bias", [1, 128])  # fully valid [1, 128]
        result_type = tile.matmul_bias(lhs, rhs, bias).type
        assert _valid_ints(result_type) == [40, 96]

    def test_matmul_bias_rejects_static_valid_k_mismatch(self):
        """matmul_bias still validates the contraction K on the valid extents."""
        lhs = _mk_tile("lhs", [128, 128], [40, 64])  # valid K = 64
        rhs = _mk_tile("rhs", [128, 128], [32, 96])  # valid K = 32
        bias = _mk_tile("bias", [1, 128])
        with pytest.raises(ValueError, match="disagree on the valid contraction length K"):
            tile.matmul_bias(lhs, rhs, bias)

    def test_matmul_acc_empty_accumulator_narrows(self):
        """acc += lhs @ rhs into a create_tile(valid=[0,0]) accumulator narrows to [M, N]."""
        acc = _mk_tile("acc", [128, 128], [0, 0])  # empty accumulator
        lhs = _mk_tile("lhs", [128, 128], [40, 64])  # valid [M=40, K=64]
        rhs = _mk_tile("rhs", [128, 128], [64, 96])  # valid [K=64, N=96]
        result_type = tile.matmul_acc(acc, lhs, rhs).type
        # Union of the empty acc [0,0] with the written [40,96] rectangle at origin.
        assert _valid_ints(result_type) == [40, 96]

    def test_matmul_acc_full_accumulator_stays_full(self):
        """A fully-valid accumulator unions back to the full shape (D2) — never narrows."""
        acc = _mk_tile("acc", [128, 128])  # fully valid
        lhs = _mk_tile("lhs", [128, 128], [40, 64])
        rhs = _mk_tile("rhs", [128, 128], [64, 96])
        result_type = tile.matmul_acc(acc, lhs, rhs).type
        assert _valid_ints(result_type) == [128, 128]

    def test_matmul_acc_grows_accumulator_valid_region(self):
        """Two accumulations into one accumulator grow the valid region monotonically."""
        acc = _mk_tile("acc", [128, 128], [40, 96])  # already valid over [40, 96]
        lhs = _mk_tile("lhs", [128, 128], [72, 64])  # M = 72 grows rows
        rhs = _mk_tile("rhs", [128, 128], [64, 96])  # N = 96 matches
        result_type = tile.matmul_acc(acc, lhs, rhs).type
        # Bounding box max([40,96], [72,96]) = [72, 96].
        assert _valid_ints(result_type) == [72, 96]

    def test_gemv_acc_empty_accumulator_narrows(self):
        """gemv_acc into an empty [1,N] accumulator narrows to [1, N-valid]."""
        acc = _mk_tile("acc", [1, 128], [0, 0])
        lhs = _mk_tile("lhs", [1, 128], [1, 64])  # valid [M=1, K=64]
        rhs = _mk_tile("rhs", [128, 128], [64, 96])  # valid [K=64, N=96]
        result_type = tile.gemv_acc(acc, lhs, rhs).type
        assert _valid_ints(result_type) == [1, 96]

    def test_gemv_bias_propagates_m_n_valid_axes(self):
        """gemv_bias out valid = [M=1, N] from lhs axis 0 / rhs axis 1."""
        lhs = _mk_tile("lhs", [1, 128], [1, 64])
        rhs = _mk_tile("rhs", [128, 128], [64, 96])
        bias = _mk_tile("bias", [1, 128])
        result_type = tile.gemv_bias(lhs, rhs, bias).type
        assert _valid_ints(result_type) == [1, 96]

    def test_matmul_bias_narrows_n_to_bias_valid(self):
        """A bias valid over fewer N cols narrows the output N: C[:,j]=A@B+bias is real
        only where bias[0,j] is real, so bias padding must NOT be claimed valid.
        """
        lhs = _mk_tile("lhs", [128, 128], [40, 64])  # valid [M=40, K=64]
        rhs = _mk_tile("rhs", [128, 128], [64, 96])  # valid [K=64, N=96]
        bias = _mk_tile("bias", [1, 128], [1, 50])  # bias valid over 50 cols only
        result_type = tile.matmul_bias(lhs, rhs, bias).type
        # N clamps to min(rhs N=96, bias N=50) = 50; the 46 bias-padded cols are invalid.
        # NOT the pre-fix [40, 96], which widened the bias-padded columns to valid.
        assert _valid_ints(result_type) == [40, 50]

    def test_gemv_bias_narrows_n_to_bias_valid(self):
        """gemv_bias narrows output N to the bias valid N, same rule as matmul_bias."""
        lhs = _mk_tile("lhs", [1, 128], [1, 64])
        rhs = _mk_tile("rhs", [128, 128], [64, 96])
        bias = _mk_tile("bias", [1, 128], [1, 50])
        result_type = tile.gemv_bias(lhs, rhs, bias).type
        assert _valid_ints(result_type) == [1, 50]

    def test_matmul_bias_fully_padding_bias_row_zeros_n(self):
        """A bias whose sole row is padding (valid rows 0) corrupts every output column,
        so the effective bias N — and thus the output N — is 0.
        """
        lhs = _mk_tile("lhs", [128, 128], [40, 64])
        rhs = _mk_tile("rhs", [128, 128], [64, 96])
        bias = _mk_tile("bias", [1, 128], [0, 96])  # the single bias row is padding
        result_type = tile.matmul_bias(lhs, rhs, bias).type
        assert _valid_ints(result_type) == [40, 0]

    def test_matmul_bias_symbolic_bias_n_defers_to_min(self):
        """A symbolic bias N cannot be folded: output N is min(rhs N, bias N) — a Min
        node — never widened back to the raw matmul N. Dynamic-bias regression guard.
        """
        bn = _index_scalar("BN")
        lhs = _mk_tile("lhs", [128, 128], [40, 64])
        rhs = _mk_tile("rhs", [128, 128], [64, 96])
        bias = _mk_tile("bias", [1, 128], [1, bn])
        result_type = tile.matmul_bias(lhs, rhs, bias).type
        assert isinstance(result_type, ir.TileType)
        assert result_type.tile_view is not None
        valid = result_type.tile_view.valid_shape
        assert isinstance(valid[0], ir.ConstInt) and valid[0].value == 40
        # N is a genuine min(96, BN): a Min node, NOT the widened ConstInt 96.
        assert isinstance(valid[1], ir.Min)


class TestTileTransformOps:
    """Test suite for tile-level transform operators."""

    def test_tile_transpose(self):
        """Test tile.transpose operator - transpose a tile."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 64], pl.FP32],
                output: pl.Tensor[[64, 128], pl.FP32],
            ) -> pl.Tensor[[64, 128], pl.FP32]:
                tile_a: pl.Tile[[32, 16], pl.FP32] = pl.load(a, [0, 0], [32, 16])
                tile_c: pl.Tile[[16, 32], pl.FP32] = pl.transpose(tile_a, axis1=0, axis2=1)
                result: pl.Tensor[[64, 128], pl.FP32] = pl.store(tile_c, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.transpose" in ir_str

    # ------------------------------------------------------------------
    # valid_shape propagation: transpose PERMUTES the input's
    # valid region rather than resetting it to the full transposed shape.
    # ------------------------------------------------------------------

    def test_tile_transpose_fully_valid_canonicalizes(self):
        """A fully-valid input transposes to a fully-valid output (no view)."""
        result_type = tile.transpose(_mk_tile("a", [64, 128]), 0, 1).type
        assert isinstance(result_type, ir.TileType)
        assert _valid_ints(result_type) == [128, 64]
        assert result_type.tile_view is None

    def test_tile_transpose_partial_permutes(self):
        """valid [32, 64] on shape [64, 128] -> valid [64, 32] on shape [128, 64]."""
        result_type = tile.transpose(_mk_tile("a", [64, 128], [32, 64]), 0, 1).type
        assert isinstance(result_type, ir.TileType)
        assert _const_ints(result_type.shape) == [128, 64]
        assert _valid_ints(result_type) == [64, 32]

    def test_tile_transpose_dynamic_symbolic(self):
        """A dynamic valid extent is carried through the permutation unchanged."""
        m = _index_scalar("M")
        result_type = tile.transpose(_mk_tile("a", [64, 128], [m, 64]), 0, 1).type
        assert isinstance(result_type, ir.TileType)
        assert result_type.tile_view is not None
        valid = result_type.tile_view.valid_shape
        # After swapping axes 0/1 the dynamic M lands on axis 1.
        assert isinstance(valid[0], ir.ConstInt) and valid[0].value == 64
        assert valid[1] is m


class TestTileSliceReshapeOps:
    """Tests for tile slice and reshape operations."""

    def test_tile_slice(self):
        """Test tile.slice operation."""
        span = ir.Span.unknown()

        # Create a tile variable [16, 32]
        dim16 = ir.ConstInt(16, DataType.INT32, span)
        dim32 = ir.ConstInt(32, DataType.INT32, span)
        tile_type = ir.TileType([dim16, dim32], DataType.FP16)
        tile_var = ir.Var("tile", tile_type, span)

        # Create a slice [8, 16] with offset [0, 0]
        call = tile.slice(tile_var, [8, 16], [0, 0])

        assert isinstance(call, ir.Call)
        assert call.op.name == "tile.slice"
        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        assert result_type.dtype == DataType.FP16
        assert len(result_type.shape) == 2

    def test_tile_slice_with_dynamic_valid_shape(self):
        """tile.slice keeps static allocation shape and stores dynamic valid_shape in TileView."""
        span = ir.Span.unknown()

        dim8 = ir.ConstInt(8, DataType.INT32, span)
        dim16 = ir.ConstInt(16, DataType.INT32, span)
        valid_n = ir.Var("valid_n", ir.ScalarType(DataType.INDEX), span)
        tile_type = ir.TileType([dim8, dim16], DataType.FP16)
        tile_var = ir.Var("tile", tile_type, span)

        call = tile.slice(tile_var, [8, 16], [0, 0], valid_shape=[8, valid_n])

        assert isinstance(call, ir.Call)
        assert call.op.name == "tile.slice"
        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        assert result_type.tile_view is not None
        assert len(result_type.shape) == 2
        assert isinstance(result_type.shape[1], ir.ConstInt)
        assert result_type.tile_view.valid_shape[1] is valid_n

    # ------------------------------------------------------------------
    # valid_shape propagation (INTERSECT): slice INTERSECTS
    # the window with the SOURCE tile's own valid region, never widening past it.
    # Gated on the source carrying an explicit region (a fully-valid source is
    # untouched — the window is all real data).
    # ------------------------------------------------------------------

    def test_tile_slice_fully_valid_source_unchanged(self):
        """A fully-valid source leaves the window's valid_shape as-is (no churn)."""
        result_type = tile.slice(_mk_tile("s", [16, 32]), [8, 16], [0, 0]).type
        assert _valid_ints(result_type) == [8, 16]

    def test_tile_slice_intersects_partial_source(self):
        """Partial src valid [6, 20] on [16, 32], slice [8, 16] @ [0, 0] ->
        row clamp(6, 0, 8) = 6 (STRICTLY narrower than the window 8), col
        clamp(20, 0, 16) = 16. The row < window makes this diverge from a
        revert-to-``new_shape`` baseline (which would give the full [8, 16])."""
        result_type = tile.slice(_mk_tile("s", [16, 32], [6, 20]), [8, 16], [0, 0]).type
        assert _valid_ints(result_type) == [6, 16]

    def test_tile_slice_intersects_partial_source_with_offset(self):
        """Partial src valid [10, 20], slice [8, 16] @ [6, 10] ->
        clamp(10 - 6, 0, 8) = 4, clamp(20 - 10, 0, 16) = 10."""
        result_type = tile.slice(_mk_tile("s", [16, 32], [10, 20]), [8, 16], [6, 10]).type
        assert _valid_ints(result_type) == [4, 10]

    def test_tile_slice_explicit_valid_intersects_source(self):
        """An explicit valid_shape= is intersected with the source region (min), never widened."""
        # src valid [10, 20]; window [8, 16] @ 0; explicit valid [5, 16]
        # -> col: min(16, clamp(20,0,16)=16) = 16 ; row: min(5, clamp(10,0,8)=8) = 5
        result_type = tile.slice(_mk_tile("s", [16, 32], [10, 20]), [8, 16], [0, 0], valid_shape=[5, 16]).type
        assert _valid_ints(result_type) == [5, 16]

    def test_tile_slice_dynamic_source_symbolic(self):
        """A dynamic offset into a partial source yields a symbolic (min/max) extent."""
        span = ir.Span.unknown()
        r = ir.Var("r", ir.ScalarType(DataType.INDEX), span)
        result_type = tile.slice(_mk_tile("s", [16, 32], [10, 20]), [8, 16], [r, 0]).type
        assert isinstance(result_type, ir.TileType)
        assert result_type.tile_view is not None
        valid = result_type.tile_view.valid_shape
        # row = min(max(10 - r, 0), 8) — a symbolic Min node, not a folded ConstInt.
        assert isinstance(valid[0], ir.Min) and not isinstance(valid[0], ir.ConstInt)
        # col offset 0 folds: clamp(20, 0, 16) = 16.
        assert isinstance(valid[1], ir.ConstInt) and valid[1].value == 16

    # ------------------------------------------------------------------
    # clamp=True: clip the window to the source tile's valid
    # region (its physical shape when unset) at the offset, even for a fully-valid
    # source. Composes with the 4d source-region intersect in a single min.
    # ------------------------------------------------------------------

    def test_tile_slice_clamp_fully_valid_source_clips_to_edge(self):
        """clamp=True clips to the physical edge of a fully-valid source:
        [16,32] slice [8,16] @ [12,0] -> row min(8, 16-12)=4, col 16."""
        result_type = tile.slice(_mk_tile("s", [16, 32]), [8, 16], [12, 0], clamp=True).type
        assert _valid_ints(result_type) == [4, 16]

    def test_tile_slice_no_clamp_fully_valid_source_unchanged(self):
        """Without clamp a fully-valid source's window is untouched (contrast to clamp)."""
        result_type = tile.slice(_mk_tile("s", [16, 32]), [8, 16], [12, 0]).type
        assert _valid_ints(result_type) == [8, 16]

    def test_tile_slice_clamp_clips_both_dims_to_edge(self):
        """clamp=True clips BOTH dims of a fully-valid source at a two-dim offset.

        [16,32] slice [8,16] @ [10,20]: row min(8, 16-10)=6, col min(16, 32-20)=12.
        Unlike the single-row-dim edge test, the col offset is non-zero so this locks
        clamp's col-dim clipping too, and it is revert-catching: without clamp a
        fully-valid source is untouched -> the full window [8,16].
        """
        result_type = tile.slice(_mk_tile("s", [16, 32]), [8, 16], [10, 20], clamp=True).type
        assert _valid_ints(result_type) == [6, 12]

    def test_tile_slice_clamp_intersects_explicit_valid(self):
        """clamp=True AND an explicit valid_shape intersect (min), with clamp BINDING.

        fully-valid src, window [8,16] @ [12,0] clamps row to 16-12 = 4; the explicit
        row 6 is LARGER, so the min is dominated by clamp's derived 4 -> the result is
        [4,16], not the explicit [6,16]. This makes the test revert-catching: dropping
        the clamp production would leave the explicit [6,16] verbatim (a fully-valid
        source has no 4d source-region intersect), so it would fail here.
        """
        result_type = tile.slice(
            _mk_tile("s", [16, 32]), [8, 16], [12, 0], valid_shape=[6, 16], clamp=True
        ).type
        assert _valid_ints(result_type) == [4, 16]

    def test_tile_slice_clamp_dynamic_offset_symbolic(self):
        """clamp=True with a dynamic offset into a fully-valid source is symbolic."""
        span = ir.Span.unknown()
        r = ir.Var("r", ir.ScalarType(DataType.INDEX), span)
        result_type = tile.slice(_mk_tile("s", [16, 32]), [8, 16], [r, 0], clamp=True).type
        assert isinstance(result_type, ir.TileType)
        assert result_type.tile_view is not None
        valid = result_type.tile_view.valid_shape
        assert isinstance(valid[0], ir.Min) and not isinstance(valid[0], ir.ConstInt)
        assert isinstance(valid[1], ir.ConstInt) and valid[1].value == 16

    def test_tile_slice_clamp_negative_offset_rejected(self):
        """A provably-negative constant offset is not origin-anchored -> reject, never widen.

        Adversary regression: clamp only clipped the HIGH edge, so a negative offset
        computed avail = src_valid - (-10) = src_valid + 10 and WIDENED the row extent
        (rows [0,10) are pre-origin, out-of-bounds padding). Both a narrow-source and a
        fully-valid source must reject rather than report a full/widened extent.
        """
        with pytest.raises(ValueError, match="offset is negative"):
            tile.slice(_mk_tile("s", [128, 128], [40, 50]), [64, 64], [-10, 0], clamp=True)
        with pytest.raises(ValueError, match="offset is negative"):
            tile.slice(_mk_tile("s", [128, 128]), [64, 64], [-10, 0], clamp=True)

    def test_tile_slice_clamp_rank_mismatch_rejected(self):
        """clamp=True on a rank-mismatched slice rejects rather than defaulting to full.

        A 1D window over a 2D tile cannot be clipped dim-for-dim; the North Star forbids
        falling through to the full-window valid_shape. Mirrors the tensor.slice guard.
        The non-clamp form of the same rank-mismatched slice is still accepted (no churn).
        """
        with pytest.raises(ValueError, match="clamp=True requires"):
            tile.slice(_mk_tile("s", [128, 128]), [64], [64], clamp=True)
        # Without clamp the rank-mismatched slice is untouched (existing 4d behavior).
        assert tile.slice(_mk_tile("s", [128, 128]), [64], [64]).op.name == "tile.slice"

    def test_tile_slice_rejects_dynamic_shape(self):
        """tile.slice shape must stay static so InitMemRef can allocate memory."""
        span = ir.Span.unknown()

        dim8 = ir.ConstInt(8, DataType.INT32, span)
        dim16 = ir.ConstInt(16, DataType.INT32, span)
        valid_n = ir.Var("valid_n", ir.ScalarType(DataType.INDEX), span)
        tile_type = ir.TileType([dim8, dim16], DataType.FP16)
        tile_var = ir.Var("tile", tile_type, span)

        with pytest.raises(ValueError, match="compile-time constant"):
            tile.slice(tile_var, [8, valid_n], [0, 0])

    def test_tile_slice_drop_dims_rank_reduces(self):
        """tile.slice drop_dims erases the listed unit axes from the result type."""
        span = ir.Span.unknown()
        tile_var = ir.Var("tile", ir.TileType([64, 64, 64, 64], DataType.FP16), span)

        call = tile.slice(tile_var, [1, 64, 64, 64], [3, 0, 0, 0], drop_dims=[0])
        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        assert [d.value for d in result_type.shape if isinstance(d, ir.ConstInt)] == [64, 64, 64]
        # shape / offset stay full-rank; drop_dims is the 5th operand.
        assert len(call.args) == 5

    def test_tile_slice_drop_dims_clamps_to_2d(self):
        """A natural sub-2D result is clamped back to 2D by prepending unit axes."""
        span = ir.Span.unknown()
        tile_var = ir.Var("tile", ir.TileType([64, 64, 64, 64], DataType.FP16), span)

        call = tile.slice(tile_var, [1, 1, 1, 64], [1, 2, 3, 0], drop_dims=[0, 1, 2])
        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        assert [d.value for d in result_type.shape if isinstance(d, ir.ConstInt)] == [1, 64]

    def test_tile_slice_drop_dims_rejects_non_unit_dim(self):
        """tile.slice drop_dims may only erase statically size-1 dimensions."""
        span = ir.Span.unknown()
        tile_var = ir.Var("tile", ir.TileType([64, 64], DataType.FP16), span)
        with pytest.raises(ValueError, match="static unit dimension"):
            tile.slice(tile_var, [8, 64], [0, 0], drop_dims=[0])

    def test_tile_slice_empty_drop_dims_is_backward_compatible(self):
        """drop_dims=None / [] keeps the legacy 3-arg behavior."""
        span = ir.Span.unknown()
        tile_var = ir.Var("tile", ir.TileType([16, 32], DataType.FP16), span)
        call_none = tile.slice(tile_var, [8, 16], [0, 0])
        call_empty = tile.slice(tile_var, [8, 16], [0, 0], drop_dims=[])
        for call in (call_none, call_empty):
            assert len(call.args) == 3
            result_type = call.type
            assert isinstance(result_type, ir.TileType)
            assert [d.value for d in result_type.shape if isinstance(d, ir.ConstInt)] == [8, 16]

    def test_tile_slice_drop_dims_print_parse_roundtrip(self):
        """A drop_dims tile slice survives python_print -> pl.parse -> python_print."""
        src = (
            "import pypto.language as pl\n\n"
            "@pl.program\n"
            "class Q:\n"
            "    @pl.function\n"
            "    def main(self, x: pl.Tile[[64, 64, 64, 64], pl.FP16]) -> pl.Tile[[1, 64], pl.FP16]:\n"
            "        y: pl.Tile[[1, 64], pl.FP16] = "
            "pl.tile.slice(x, [1, 1, 1, 64], [1, 2, 3, 0], drop_dims=[0, 1, 2])\n"
            "        return y\n"
        )
        prog = pl.parse(src)
        reparsed = pl.parse(ir.python_print(prog))
        ir.assert_structural_equal(reparsed, prog)

    @staticmethod
    def _make_slice_tile_var():
        """Build a [16, 32] FP16 tile Var for slice pad_value tests."""
        span = ir.Span.unknown()
        dim16 = ir.ConstInt(16, DataType.INT32, span)
        dim32 = ir.ConstInt(32, DataType.INT32, span)
        tile_type = ir.TileType([dim16, dim32], DataType.FP16)
        return ir.Var("tile", tile_type, span)

    def test_tile_slice_with_pad_value_zero(self):
        """tile.slice writes pad_value=zero to the output tile_view.pad."""
        tile_var = self._make_slice_tile_var()
        call = tile.slice(tile_var, [8, 16], [0, 0], valid_shape=[8, 4], pad_value=ir.PadValue.zero)

        assert isinstance(call, ir.Call)
        assert call.op.name == "tile.slice"
        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        assert result_type.tile_view is not None
        assert result_type.tile_view.pad == ir.PadValue.zero
        assert len(result_type.tile_view.valid_shape) == 2
        assert isinstance(result_type.tile_view.valid_shape[0], ir.ConstInt)
        assert result_type.tile_view.valid_shape[0].value == 8
        assert isinstance(result_type.tile_view.valid_shape[1], ir.ConstInt)
        assert result_type.tile_view.valid_shape[1].value == 4

    def test_tile_slice_with_pad_value_min(self):
        """tile.slice writes pad_value=min to the output tile_view.pad."""
        tile_var = self._make_slice_tile_var()
        call = tile.slice(tile_var, [8, 16], [0, 0], valid_shape=[8, 4], pad_value=ir.PadValue.min)

        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        assert result_type.tile_view is not None
        assert result_type.tile_view.pad == ir.PadValue.min

    def test_tile_slice_with_pad_value_max(self):
        """tile.slice writes pad_value=max to the output tile_view.pad."""
        tile_var = self._make_slice_tile_var()
        call = tile.slice(tile_var, [8, 16], [0, 0], valid_shape=[8, 4], pad_value=ir.PadValue.max)

        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        assert result_type.tile_view is not None
        assert result_type.tile_view.pad == ir.PadValue.max

    def test_tile_slice_default_pad_is_null(self):
        """tile.slice without pad_value defaults to PadValue.null (backward compat)."""
        tile_var = self._make_slice_tile_var()
        call = tile.slice(tile_var, [8, 16], [0, 0])

        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        assert result_type.get_effective_tile_view().pad == ir.PadValue.null

    def test_tile_slice_rejects_bad_pad_value(self):
        """tile.slice rejects a non-PadValue pad_value kwarg via registry validation."""
        tile_var = self._make_slice_tile_var()
        span = tile_var.span
        shape_tuple = ir.MakeTuple(
            [ir.ConstInt(8, DataType.INT32, span), ir.ConstInt(16, DataType.INT32, span)], span
        )
        offset_tuple = ir.MakeTuple(
            [ir.ConstInt(0, DataType.INT32, span), ir.ConstInt(0, DataType.INT32, span)], span
        )
        valid_shape_tuple = ir.MakeTuple(
            [ir.ConstInt(8, DataType.INT32, span), ir.ConstInt(4, DataType.INT32, span)], span
        )
        with pytest.raises(TypeError, match="'pad_value'.*incompatible type"):
            ir.create_op_call(
                "tile.slice",
                [tile_var, shape_tuple, offset_tuple, valid_shape_tuple],
                {"pad_value": 5},
                span,
            )

    def test_tile_slice_accepts_numeric_sugar_pad_value(self):
        """tile.slice maps 0 / math.inf / -math.inf onto PadValue zero/max/min."""
        tile_var = self._make_slice_tile_var()
        for literal, expected_pad in [
            (0, ir.PadValue.zero),
            (math.inf, ir.PadValue.max),
            (-math.inf, ir.PadValue.min),
        ]:
            call = tile.slice(tile_var, [8, 16], [0, 0], valid_shape=[8, 4], pad_value=literal)
            result_type = call.type
            assert isinstance(result_type, ir.TileType)
            assert result_type.tile_view is not None
            assert result_type.tile_view.pad == expected_pad

    def test_tile_slice_rejects_bad_numeric_pad_value_at_python_boundary(self):
        """Non-sugar numeric values are rejected at the Python API boundary."""
        tile_var = self._make_slice_tile_var()
        with pytest.raises(ValueError, match="fillpad pad_value"):
            tile.slice(tile_var, [8, 16], [0, 0], valid_shape=[8, 4], pad_value=5)

    def test_tile_slice_pad_without_valid_shape_warns(self):
        """DSL emits a UserWarning when pad_value is set but valid_shape is None."""
        span = ir.Span.unknown()
        dim16 = ir.ConstInt(16, DataType.INT32, span)
        dim32 = ir.ConstInt(32, DataType.INT32, span)
        tile_type = ir.TileType([dim16, dim32], DataType.FP16)
        tile_var = ir.Var("tile", tile_type, span)

        tile_arg = pl.Tile(expr=tile_var)
        with pytest.warns(UserWarning, match="pad_value has no effect"):
            pl.tile.slice(tile_arg, [8, 16], [0, 0], pad_value=pl.PadValue.zero)

    def test_tile_reshape(self):
        """Test tile.reshape operation."""
        span = ir.Span.unknown()

        # Create a tile variable [4, 8]
        dim4 = ir.ConstInt(4, DataType.INT32, span)
        dim8 = ir.ConstInt(8, DataType.INT32, span)
        tile_type = ir.TileType([dim4, dim8], DataType.FP32)
        tile_var = ir.Var("tile", tile_type, span)

        # Reshape to [8, 4]
        call = tile.reshape(tile_var, [8, 4])

        assert isinstance(call, ir.Call)
        assert call.op.name == "tile.reshape"
        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        assert result_type.dtype == DataType.FP32
        assert len(result_type.shape) == 2

        # Reshape to [32, 1] — [N,1] infers col_major blayout.
        call2 = tile.reshape(tile_var, [32, 1])
        result_type2 = call2.type
        assert isinstance(result_type2, ir.TileType)
        assert len(result_type2.shape) == 2
        assert result_type2.get_effective_tile_view().blayout == ir.TileLayout.col_major

        # Layout is inferred from target shape for vector repair
        call3 = tile.reshape(tile_var, [1, 32])
        result_type3 = call3.type
        assert isinstance(result_type3, ir.TileType)
        assert result_type3.get_effective_tile_view().blayout == ir.TileLayout.row_major
        assert call3.kwargs == {}

    # ------------------------------------------------------------------
    # valid_shape propagation: reshape is only representable for a
    # partially-valid input when the valid region is a contiguous flat prefix that
    # stays a rectangle under the new shape (same precondition as FlattenTileNdTo2d).
    # A fully-valid input always reshapes to a fully-valid output.
    # ------------------------------------------------------------------

    def test_tile_reshape_fully_valid_regression(self):
        """Regression guard: a fully-valid reshape still works and stays fully valid."""
        result_type = tile.reshape(_mk_tile("a", [4, 8]), [8, 4]).type
        assert isinstance(result_type, ir.TileType)
        assert _valid_ints(result_type) == [8, 4]
        assert result_type.tile_view is None  # fully valid canonicalizes to no view

    def test_tile_reshape_partial_prefix_narrows(self):
        """valid [2, 8] on [4, 8] is the flat prefix [0, 16); reshaped to [8, 4] -> [4, 4]."""
        result_type = tile.reshape(_mk_tile("a", [4, 8], [2, 8]), [8, 4]).type
        assert _valid_ints(result_type) == [4, 4]

    def test_tile_reshape_prefix_into_1d(self):
        """valid [3, 4] on [4, 4] is the prefix [0, 12); reshaped to [16] -> [12]."""
        result_type = tile.reshape(_mk_tile("a", [4, 4], [3, 4]), [16]).type
        assert _valid_ints(result_type) == [12]

    def test_tile_reshape_vector_repair_static(self):
        """[N, 1] valid [5, 1] -> [1, N] valid [1, 5] (column<->row vector repair)."""
        result_type = tile.reshape(_mk_tile("a", [8, 1], [5, 1]), [1, 8]).type
        assert _valid_ints(result_type) == [1, 5]

    def test_tile_reshape_vector_repair_dynamic(self):
        """[N, 1] valid [vn, 1] -> [1, N] valid [1, vn] with a dynamic free extent."""
        vn = _index_scalar("vn")
        result_type = tile.reshape(_mk_tile("a", [8, 1], [vn, 1]), [1, 8]).type
        assert isinstance(result_type, ir.TileType)
        assert result_type.tile_view is not None
        valid = result_type.tile_view.valid_shape
        assert isinstance(valid[0], ir.ConstInt) and valid[0].value == 1
        assert valid[1] is vn

    def test_tile_reshape_rejects_non_prefix(self):
        """valid [4, 2] on [4, 8] is a STRIDED region (dim1 partial below a non-unit dim);
        it is not a contiguous flat prefix, so reshaping it is rejected, not widened."""
        with pytest.raises(ValueError, match="not a contiguous prefix"):
            tile.reshape(_mk_tile("a", [4, 8], [4, 2]), [8, 4])

    def test_tile_reshape_rejects_unaligned_prefix(self):
        """A valid prefix that does not align to a rectangular sub-region is rejected."""
        # valid [3, 4] on [4, 4] -> prefix [0, 12); into [8, 2]: 12 -> [6, 2] is fine, but
        # into [5, ...] would not divide. Use [4,4] valid [3,4] -> [16] is fine; craft a
        # genuinely unaligned case: prefix 12 into shape [16] with a middle boundary.
        # [8, 2] valid [3, 2] -> prefix [0, 6); into [4, 4]: 6 % 4 = 2, 6 % 1 = 0 -> w=6 > 4
        # -> no aligned split -> reject.
        with pytest.raises(ValueError, match="does not align to a rectangular sub-region"):
            tile.reshape(_mk_tile("a", [8, 2], [3, 2]), [4, 4])

    def test_tile_reshape_rejects_dynamic_target_dim(self):
        """A partially-valid input cannot be reshaped onto a target with a non-const
        shape dim: the prefix cannot be placed as a static box, so it is rejected
        rather than widened. valid [2,8] on [4,8] (prefix [0,16)) into [N, 4]."""
        span = ir.Span.unknown()
        n = ir.Var("N", ir.ScalarType(DataType.INDEX), span)
        with pytest.raises(ValueError, match="cannot map a partially-valid tile"):
            tile.reshape(_mk_tile("a", [4, 8], [2, 8]), [n, 4])

    def test_tile_reshape_rejects_unprovable_dynamic_prefix(self):
        """A symbolic free extent whose trailing sub-volume matches no target suffix
        product cannot be proven to reshape to a rectangle: valid [vn,8] on [4,8]
        (trailing sub-volume 8) into [8,4] — neither target suffix (1, 4) equals 8."""
        span = ir.Span.unknown()
        vn = ir.Var("vn", ir.ScalarType(DataType.INDEX), span)
        with pytest.raises(ValueError, match="cannot prove the dynamic valid region"):
            tile.reshape(_mk_tile("a", [4, 8], [vn, 8]), [8, 4])

    def test_tile_fillpad_expand(self):
        """Test tile.fillpad_expand grows the tile and fills with pad_value."""
        span = ir.Span.unknown()

        # Source tile [48, 64], valid [40, 50].
        dim48 = ir.ConstInt(48, DataType.INT32, span)
        dim64 = ir.ConstInt(64, DataType.INT32, span)
        src_type = ir.TileType([dim48, dim64], DataType.FP32)
        src = ir.Var("src", src_type, span)

        # Expand to [64, 128] with zero padding.
        call = tile.fillpad_expand(src, [64, 128], pad_value=ir.PadValue.zero)

        assert isinstance(call, ir.Call)
        assert call.op.name == "tile.fillpad_expand"
        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        assert result_type.dtype == DataType.FP32
        # Output physical shape is the requested (larger) shape.
        rows, cols = result_type.shape[0], result_type.shape[1]
        assert isinstance(rows, ir.ConstInt)
        assert isinstance(cols, ir.ConstInt)
        assert rows.value == 64
        assert cols.value == 128
        view = result_type.get_effective_tile_view()
        # After expand the whole destination is valid and carries the pad mode.
        assert view.pad == ir.PadValue.zero
        vrows, vcols = view.valid_shape[0], view.valid_shape[1]
        assert isinstance(vrows, ir.ConstInt)
        assert isinstance(vcols, ir.ConstInt)
        assert vrows.value == 64
        assert vcols.value == 128

        # max / min pad modes round-trip onto the result view.
        call_max = tile.fillpad_expand(src, [64, 128], pad_value=ir.PadValue.max)
        max_type = call_max.type
        assert isinstance(max_type, ir.TileType)
        assert max_type.get_effective_tile_view().pad == ir.PadValue.max

    def test_tile_fillpad_expand_same_shape(self):
        """tile.fillpad_expand permits a same-shape (non-strict) expansion."""
        span = ir.Span.unknown()
        dim32 = ir.ConstInt(32, DataType.INT32, span)
        src_type = ir.TileType([dim32, dim32], DataType.FP16)
        src = ir.Var("src", src_type, span)

        call = tile.fillpad_expand(src, [32, 32], pad_value=ir.PadValue.zero)
        assert call.op.name == "tile.fillpad_expand"
        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        dim0 = result_type.shape[0]
        assert isinstance(dim0, ir.ConstInt)
        assert dim0.value == 32

    def test_tile_fillpad_expand_shrink_raises(self):
        """tile.fillpad_expand rejects a destination smaller than the source."""
        span = ir.Span.unknown()
        dim64 = ir.ConstInt(64, DataType.INT32, span)
        src_type = ir.TileType([dim64, dim64], DataType.FP32)
        src = ir.Var("src", src_type, span)

        with pytest.raises(ValueError, match="must be >= source dimension"):
            tile.fillpad_expand(src, [32, 64], pad_value=ir.PadValue.zero)

    def test_tile_fillpad_expand_program(self):
        """tile.fillpad_expand is reachable from the DSL and prints in the IR."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[48, 64], pl.FP32],
                output: pl.Tensor[[64, 64], pl.FP32],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                src: pl.Tile[[48, 64], pl.FP32] = pl.load(a, [0, 0], [48, 64])
                dst: pl.Tile[[64, 64], pl.FP32] = pl.tile.fillpad_expand(
                    src, [64, 64], pad_value=pl.PadValue.zero
                )
                result: pl.Tensor[[64, 64], pl.FP32] = pl.store(dst, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.fillpad_expand" in ir_str

    def test_tile_transpose(self):
        """Test tile.transpose operation."""
        span = ir.Span.unknown()

        # Create a tile [8, 16]
        dim8 = ir.ConstInt(8, DataType.INT32, span)
        dim16 = ir.ConstInt(16, DataType.INT32, span)
        tile_type = ir.TileType([dim8, dim16], DataType.FP16)
        tile_var = ir.Var("tile", tile_type, span)

        # Transpose: [8, 16] -> [16, 8]
        call = tile.transpose(tile_var, 0, 1)

        assert isinstance(call, ir.Call)
        assert call.op.name == "tile.transpose"
        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        assert result_type.dtype == DataType.FP16
        assert len(result_type.shape) == 2

    def test_tile_transpose_negative_axis(self):
        """Test tile.transpose with negative axis indices."""
        span = ir.Span.unknown()

        # Create a tile [8, 16]
        dim8 = ir.ConstInt(8, DataType.INT32, span)
        dim16 = ir.ConstInt(16, DataType.INT32, span)
        tile_type = ir.TileType([dim8, dim16], DataType.FP32)
        tile_var = ir.Var("tile", tile_type, span)

        # Transpose using negative indices: axis1=-2 (0), axis2=-1 (1)
        # [8, 16] -> [16, 8]
        call = tile.transpose(tile_var, -2, -1)

        assert isinstance(call, ir.Call)
        assert call.op.name == "tile.transpose"
        result_type = call.type
        assert isinstance(result_type, ir.TileType)

    def test_tile_transpose_no_auto_tmp(self):
        """tile.transpose emits the 3-arg form (no scratch) when tmp is omitted.

        The pto.ttrans scratch is materialized later by FlattenTileNdTo2D, not here.
        The optional tmp operand is only for round-tripping that lowered 4-arg form.
        """
        span = ir.Span.unknown()

        dim8 = ir.ConstInt(8, DataType.INT32, span)
        dim16 = ir.ConstInt(16, DataType.INT32, span)
        tile_type = ir.TileType([dim8, dim16], DataType.FP16)
        tile_var = ir.Var("tile", tile_type, span)

        # Omitted tmp -> 3-arg, no auto-created scratch.
        call = tile.transpose(tile_var, 0, 1)
        assert len(call.args) == 3

        # Explicit tmp (as the lowered form carries) -> 4-arg, passed through verbatim.
        tmp_var = ir.Var("tmp", tile_type, span)
        call4 = tile.transpose(tile_var, 0, 1, tmp=tmp_var)
        assert len(call4.args) == 4
        assert call4.args[3] is tmp_var

    def test_tile_set_validshape(self):
        """Test tile.set_validshape with constant valid dimensions."""
        span = ir.Span.unknown()

        dim32 = ir.ConstInt(32, DataType.INT32, span)
        tile_type = ir.TileType([dim32, dim32], DataType.FP32)
        tile_var = ir.Var("tile", tile_type, span)

        call = tile.set_validshape(tile_var, 16, 24)

        assert isinstance(call, ir.Call)
        assert call.op.name == "tile.set_validshape"
        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        assert result_type.dtype == DataType.FP32
        assert len(result_type.shape) == 2
        assert result_type.tile_view is not None
        assert len(result_type.tile_view.valid_shape) == 2

    def test_tile_set_validshape_dynamic(self):
        """Test tile.set_validshape with dynamic Scalar[INDEX] dimensions."""
        span = ir.Span.unknown()

        dim32 = ir.ConstInt(32, DataType.INT32, span)
        tile_type = ir.TileType([dim32, dim32], DataType.FP32)
        tile_var = ir.Var("tile", tile_type, span)
        valid_rows = ir.Var("vr", ir.ScalarType(DataType.INDEX), span)
        valid_cols = ir.Var("vc", ir.ScalarType(DataType.INDEX), span)

        call = tile.set_validshape(tile_var, valid_rows, valid_cols)

        assert isinstance(call, ir.Call)
        assert call.op.name == "tile.set_validshape"
        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        assert result_type.tile_view is not None
        assert result_type.tile_view.valid_shape[0] is valid_rows
        assert result_type.tile_view.valid_shape[1] is valid_cols

    def test_tile_set_validshape_keeps_implicit_acc_layout(self):
        """The result aliases the source buffer, so it must keep the source's layout.

        An Acc tile that leaves `tile_view` implicit still *has* a layout: the one
        its memory space implies (col_major / row_major / fractal=1024). Seeding the
        result's TileView from a default-constructed one would pin the raw
        row_major / none_box / fractal=512 defaults onto an alias of an Acc
        accumulator, and codegen would then annotate the shared tile_buf handle with
        a layout its own `pto.alloc_tile` never declared.
        """
        span = ir.Span.unknown()
        rows = ir.ConstInt(16, DataType.INT32, span)
        cols = ir.ConstInt(128, DataType.INT32, span)

        # Implicit tile_view (None) + Acc memory space.
        acc_type = ir.TileType([rows, cols], DataType.FP32, None, None, ir.MemorySpace.Acc)
        acc_var = ir.Var("acc", acc_type, span)
        assert acc_type.tile_view is None

        result_type = tile.set_validshape(acc_var, 5, 128).type

        assert isinstance(result_type, ir.TileType)
        # Narrowing valid_shape must not disturb the other metadata of the aliased buffer.
        assert result_type.memory_space == ir.MemorySpace.Acc
        view = result_type.tile_view
        assert view is not None
        assert view.blayout == ir.TileLayout.col_major
        assert view.slayout == ir.TileLayout.row_major
        assert view.fractal == 1024
        assert _const_values(view.valid_shape) == [5, 128]

    def test_tile_set_validshape_keeps_explicit_source_layout(self):
        """An explicit source TileView is carried through unchanged but for valid_shape."""
        span = ir.Span.unknown()
        rows = ir.ConstInt(16, DataType.INT32, span)
        cols = ir.ConstInt(128, DataType.INT32, span)

        source_view = ir.TileView(
            [rows, cols], [], None, ir.TileLayout.col_major, ir.TileLayout.col_major, 512
        )
        src_type = ir.TileType([rows, cols], DataType.FP32, None, source_view, ir.MemorySpace.Right)
        src_var = ir.Var("rhs", src_type, span)

        result_type = tile.set_validshape(src_var, 5, 128).type

        assert isinstance(result_type, ir.TileType)
        assert result_type.memory_space == ir.MemorySpace.Right
        view = result_type.tile_view
        assert view is not None
        assert view.blayout == ir.TileLayout.col_major
        assert view.slayout == ir.TileLayout.col_major
        assert view.fractal == 512
        assert _const_values(view.valid_shape) == [5, 128]

    def test_tile_set_validshape_preserves_physical_shape(self):
        """Physical shape is unchanged; only valid_shape metadata is updated."""
        span = ir.Span.unknown()

        dim16 = ir.ConstInt(16, DataType.INT32, span)
        dim64 = ir.ConstInt(64, DataType.INT32, span)
        tile_type = ir.TileType([dim16, dim64], DataType.FP16)
        tile_var = ir.Var("tile", tile_type, span)

        call = tile.set_validshape(tile_var, 8, 32)
        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        assert isinstance(result_type.shape[0], ir.ConstInt)
        assert result_type.shape[0].value == 16
        assert isinstance(result_type.shape[1], ir.ConstInt)
        assert result_type.shape[1].value == 64

    def test_tile_set_validshape_rejects_negative(self):
        """Negative constant valid dimensions are rejected."""
        span = ir.Span.unknown()

        dim16 = ir.ConstInt(16, DataType.INT32, span)
        tile_type = ir.TileType([dim16, dim16], DataType.FP32)
        tile_var = ir.Var("tile", tile_type, span)

        with pytest.raises(Exception, match="must be >= 0"):
            tile.set_validshape(tile_var, -1, 8)

    def test_tile_set_validshape_rejects_exceeding_bound(self):
        """Valid dimensions exceeding physical shape are rejected."""
        span = ir.Span.unknown()

        dim16 = ir.ConstInt(16, DataType.INT32, span)
        tile_type = ir.TileType([dim16, dim16], DataType.FP32)
        tile_var = ir.Var("tile", tile_type, span)

        with pytest.raises(Exception, match="exceeds tile bound"):
            tile.set_validshape(tile_var, 32, 8)

    def test_transform_operators_registered(self):
        """Test that transform operators are registered."""
        assert ir.is_op_registered("tile.slice")
        assert ir.is_op_registered("tile.reshape")
        assert ir.is_op_registered("tile.transpose")
        assert ir.is_op_registered("tile.set_validshape")


def _const_dims(span, *values):
    """Build a list of ConstInt dims (INT32) from Python ints."""
    return [ir.ConstInt(v, DataType.INT32, span) for v in values]


def _const_values(dims):
    """Extract the ints from a dim list, asserting every dim is a ConstInt."""
    consts = [dim for dim in dims if isinstance(dim, ir.ConstInt)]
    assert len(consts) == len(dims), f"expected all-constant dims, got {dims}"
    return [dim.value for dim in consts]


class TestTileBatchMatMulOps:
    """Tests for tile batch matrix multiplication operations."""

    @pytest.mark.parametrize(
        ("lhs_shape", "rhs_shape", "input_dtype", "expected_shape"),
        [
            # 2D: [16,32] @ [32,64] -> [16,64] (regular matmul)
            ([16, 32], [32, 64], DataType.FP16, [16, 64]),
            # 3D: [4,16,32] @ [4,32,64] -> [4,16,64] (one batch dim)
            ([4, 16, 32], [4, 32, 64], DataType.FP32, [4, 16, 64]),
            # 4D: [2,3,16,32] @ [2,3,32,64] -> [2,3,16,64] (multiple batch dims, FP16 in)
            ([2, 3, 16, 32], [2, 3, 32, 64], DataType.FP16, [2, 3, 16, 64]),
            # Broadcast: [1,16,32] @ [4,32,64] -> [4,16,64]
            ([1, 16, 32], [4, 32, 64], DataType.FP32, [4, 16, 64]),
        ],
        ids=["2d", "3d", "4d", "broadcast"],
    )
    def test_batch_matmul(self, lhs_shape, rhs_shape, input_dtype, expected_shape):
        """tile.batch_matmul handles batch ranks + broadcasting; result dtype is promoted to FP32."""
        span = ir.Span.unknown()
        lhs_type = ir.TileType(_const_dims(span, *lhs_shape), input_dtype)
        rhs_type = ir.TileType(_const_dims(span, *rhs_shape), input_dtype)
        lhs = ir.Var("lhs", lhs_type, span)
        rhs = ir.Var("rhs", rhs_type, span)

        call = tile.batch_matmul(lhs, rhs, span)

        assert isinstance(call, ir.Call)
        assert call.op.name == "tile.batch_matmul"
        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        const_dims = [dim for dim in result_type.shape if isinstance(dim, ir.ConstInt)]
        assert len(const_dims) == len(result_type.shape)
        assert [dim.value for dim in const_dims] == expected_shape
        assert result_type.dtype == DataType.FP32

    def test_batch_matmul_dtype_mismatch(self):
        """Test tile.batch_matmul rejects mismatched dtypes."""
        span = ir.Span.unknown()

        dim4 = ir.ConstInt(4, DataType.INT32, span)
        dim16 = ir.ConstInt(16, DataType.INT32, span)
        dim32 = ir.ConstInt(32, DataType.INT32, span)

        lhs_type = ir.TileType([dim4, dim16, dim32], DataType.FP16)
        rhs_type = ir.TileType([dim4, dim32, dim16], DataType.FP32)

        lhs = ir.Var("lhs", lhs_type, span)
        rhs = ir.Var("rhs", rhs_type, span)

        with pytest.raises(ValueError, match="identical"):
            tile.batch_matmul(lhs, rhs, span)

    def test_batch_matmul_int_accumulation(self):
        """Test tile.batch_matmul with integer inputs produces INT32 accumulator dtype."""
        span = ir.Span.unknown()

        dim2 = ir.ConstInt(2, DataType.INT32, span)
        dim16 = ir.ConstInt(16, DataType.INT32, span)
        dim32 = ir.ConstInt(32, DataType.INT32, span)

        lhs_type = ir.TileType([dim2, dim16, dim32], DataType.INT8)
        rhs_type = ir.TileType([dim2, dim32, dim16], DataType.INT8)

        lhs = ir.Var("lhs", lhs_type, span)
        rhs = ir.Var("rhs", rhs_type, span)

        call = tile.batch_matmul(lhs, rhs, span)

        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        assert result_type.dtype == DataType.INT32

    def test_batch_matmul_output_tile_view(self):
        """Test tile.batch_matmul output has correct TileView (col_major, row_major, fractal=1024)."""
        span = ir.Span.unknown()

        dim2 = ir.ConstInt(2, DataType.INT32, span)
        dim16 = ir.ConstInt(16, DataType.INT32, span)
        dim32 = ir.ConstInt(32, DataType.INT32, span)
        dim64 = ir.ConstInt(64, DataType.INT32, span)

        lhs_type = ir.TileType([dim2, dim16, dim32], DataType.FP16)
        rhs_type = ir.TileType([dim2, dim32, dim64], DataType.FP16)

        lhs = ir.Var("lhs", lhs_type, span)
        rhs = ir.Var("rhs", rhs_type, span)

        call = tile.batch_matmul(lhs, rhs, span)

        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        eff = result_type.get_effective_tile_view()
        assert eff.blayout == ir.TileLayout.col_major
        assert eff.slayout == ir.TileLayout.row_major
        assert eff.fractal == 1024

    @pytest.mark.parametrize(
        ("acc_shape", "lhs_shape", "rhs_shape", "input_dtype", "acc_dtype"),
        [
            # 2D: acc[16,64] += lhs[16,32] @ rhs[32,64]
            ([16, 64], [16, 32], [32, 64], DataType.FP16, DataType.FP32),
            # 3D: acc[4,16,64] += lhs[4,16,32] @ rhs[4,32,64]
            ([4, 16, 64], [4, 16, 32], [4, 32, 64], DataType.FP32, DataType.FP32),
            # 4D: multiple batch dims
            ([2, 3, 16, 64], [2, 3, 16, 32], [2, 3, 32, 64], DataType.FP16, DataType.FP32),
            # Broadcast lhs/rhs against acc batch
            ([4, 16, 64], [1, 16, 32], [4, 32, 64], DataType.FP32, DataType.FP32),
            # INT path
            ([2, 16, 64], [2, 16, 32], [2, 32, 64], DataType.INT8, DataType.INT32),
        ],
        ids=["2d", "3d", "4d", "broadcast", "int"],
    )
    def test_batch_matmul_acc(self, acc_shape, lhs_shape, rhs_shape, input_dtype, acc_dtype):
        """tile.batch_matmul_acc handles batch ranks + broadcasting; result shape == acc shape."""
        span = ir.Span.unknown()
        acc_type = ir.TileType(_const_dims(span, *acc_shape), acc_dtype)
        lhs_type = ir.TileType(_const_dims(span, *lhs_shape), input_dtype)
        rhs_type = ir.TileType(_const_dims(span, *rhs_shape), input_dtype)
        acc = ir.Var("acc", acc_type, span)
        lhs = ir.Var("lhs", lhs_type, span)
        rhs = ir.Var("rhs", rhs_type, span)

        call = tile.batch_matmul_acc(acc, lhs, rhs, span)

        assert isinstance(call, ir.Call)
        assert call.op.name == "tile.batch_matmul_acc"
        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        const_dims = [dim for dim in result_type.shape if isinstance(dim, ir.ConstInt)]
        assert len(const_dims) == len(result_type.shape)
        assert [dim.value for dim in const_dims] == acc_shape
        assert result_type.dtype == acc_dtype

    def test_batch_matmul_acc_acc_batch_must_match_broadcast(self):
        """tile.batch_matmul_acc rejects acc batch dims that disagree with broadcast(lhs, rhs)."""
        span = ir.Span.unknown()
        acc_type = ir.TileType(_const_dims(span, 2, 16, 64), DataType.FP32)
        lhs_type = ir.TileType(_const_dims(span, 4, 16, 32), DataType.FP16)
        rhs_type = ir.TileType(_const_dims(span, 4, 32, 64), DataType.FP16)
        acc = ir.Var("acc", acc_type, span)
        lhs = ir.Var("lhs", lhs_type, span)
        rhs = ir.Var("rhs", rhs_type, span)

        with pytest.raises(ValueError, match="acc batch dim"):
            tile.batch_matmul_acc(acc, lhs, rhs, span)

    def test_batch_matmul_acc_dtype_mismatch(self):
        """tile.batch_matmul_acc rejects acc dtype that doesn't match the result dtype."""
        span = ir.Span.unknown()
        # FP inputs => FP32 acc required, but acc is FP16 here.
        acc_type = ir.TileType(_const_dims(span, 2, 16, 64), DataType.FP16)
        lhs_type = ir.TileType(_const_dims(span, 2, 16, 32), DataType.FP16)
        rhs_type = ir.TileType(_const_dims(span, 2, 32, 64), DataType.FP16)
        acc = ir.Var("acc", acc_type, span)
        lhs = ir.Var("lhs", lhs_type, span)
        rhs = ir.Var("rhs", rhs_type, span)

        with pytest.raises(ValueError, match="accumulator dtype"):
            tile.batch_matmul_acc(acc, lhs, rhs, span)

    def test_batch_matmul_acc_inner_dim_mismatch(self):
        """tile.batch_matmul_acc rejects mismatched K dims."""
        span = ir.Span.unknown()
        acc_type = ir.TileType(_const_dims(span, 2, 16, 64), DataType.FP32)
        lhs_type = ir.TileType(_const_dims(span, 2, 16, 32), DataType.FP16)
        rhs_type = ir.TileType(_const_dims(span, 2, 16, 64), DataType.FP16)  # K=16, mismatch
        acc = ir.Var("acc", acc_type, span)
        lhs = ir.Var("lhs", lhs_type, span)
        rhs = ir.Var("rhs", rhs_type, span)

        with pytest.raises(ValueError, match="inner dimensions"):
            tile.batch_matmul_acc(acc, lhs, rhs, span)

    # ------------------------------------------------------------------
    # valid_shape propagation: batch dims' valid extents
    # must match between operands (modulo broadcast) and propagate to the output;
    # the trailing matrix dims carry M / N per the 2D rule.
    # ------------------------------------------------------------------

    def test_batch_matmul_propagates_batch_and_mn_valid(self):
        """[2,M,K] @ [2,K,N]: batch valid + M (lhs axis -2) + N (rhs axis -1) propagate."""
        lhs = _mk_tile("lhs", [2, 128, 128], [2, 40, 64])  # valid [B=2, M=40, K=64]
        rhs = _mk_tile("rhs", [2, 128, 128], [2, 64, 96])  # valid [B=2, K=64, N=96]
        result_type = tile.batch_matmul(lhs, rhs).type
        assert _valid_ints(result_type) == [2, 40, 96]

    def test_batch_matmul_narrowed_batch_valid_propagates(self):
        """A batch dim narrowed below the physical extent must not widen back to full."""
        lhs = _mk_tile("lhs", [4, 128, 128], [3, 40, 64])  # only 3 of 4 batches valid
        rhs = _mk_tile("rhs", [4, 128, 128], [3, 64, 96])
        result_type = tile.batch_matmul(lhs, rhs).type
        assert _valid_ints(result_type) == [3, 40, 96]

    def test_batch_matmul_broadcast_batch_valid(self):
        """A size-1 (broadcast) batch operand contributes its counterpart's valid extent."""
        lhs = _mk_tile("lhs", [1, 128, 128], [1, 40, 64])  # broadcasts across batch
        rhs = _mk_tile("rhs", [3, 128, 128], [3, 64, 96])  # 3 valid batches
        result_type = tile.batch_matmul(lhs, rhs).type
        assert _valid_ints(result_type) == [3, 40, 96]

    def test_batch_matmul_rejects_batch_valid_mismatch(self):
        """Two full-extent batch dims with a provable valid mismatch is a user error."""
        lhs = _mk_tile("lhs", [4, 128, 128], [3, 40, 64])  # 3 valid batches
        rhs = _mk_tile("rhs", [4, 128, 128], [2, 64, 96])  # 2 valid batches
        with pytest.raises(ValueError, match="valid extent of a non-broadcast batch"):
            tile.batch_matmul(lhs, rhs)

    def test_batch_matmul_rejects_static_valid_k_mismatch(self):
        """batch_matmul validates the contraction K on the valid (trailing) extents."""
        lhs = _mk_tile("lhs", [2, 128, 128], [2, 40, 64])  # valid K = 64
        rhs = _mk_tile("rhs", [2, 128, 128], [2, 32, 96])  # valid K = 32
        with pytest.raises(ValueError, match="disagree on the valid contraction length K"):
            tile.batch_matmul(lhs, rhs)

    def test_batch_matmul_acc_empty_accumulator_narrows(self):
        """batch_matmul_acc into an empty accumulator narrows to [batch, M, N]."""
        acc = _mk_tile("acc", [2, 128, 128], [0, 0, 0])  # empty accumulator
        lhs = _mk_tile("lhs", [2, 128, 128], [2, 40, 64])
        rhs = _mk_tile("rhs", [2, 128, 128], [2, 64, 96])
        result_type = tile.batch_matmul_acc(acc, lhs, rhs).type
        assert _valid_ints(result_type) == [2, 40, 96]

    def test_batch_matmul_broadcast_padding_source_zeros_batch(self):
        """A size-1 broadcast operand whose sole batch matrix is padding (valid 0) must
        NOT push the counterpart's valid batch count — every replicated output batch is
        derived from padding, so the output batch valid is 0 (never widened).
        """
        lhs = _mk_tile("lhs", [1, 128, 128], [0, 40, 64])  # sole lhs batch is padding
        rhs = _mk_tile("rhs", [3, 128, 128], [3, 64, 96])
        result_type = tile.batch_matmul(lhs, rhs).type
        assert _valid_ints(result_type) == [0, 40, 96]

    def test_batch_matmul_symbolic_batch_valid_propagates(self):
        """A symbolic batch valid extent on both operands propagates unchanged (the
        non-broadcast symbolic-defer arm): the exact Var must survive, not be dropped.
        """
        b = _index_scalar("B")
        lhs = _mk_tile("lhs", [4, 128, 128], [b, 40, 64])
        rhs = _mk_tile("rhs", [4, 128, 128], [b, 64, 96])
        result_type = tile.batch_matmul(lhs, rhs).type
        assert isinstance(result_type, ir.TileType)
        assert result_type.tile_view is not None
        valid = result_type.tile_view.valid_shape
        assert valid[0] is b  # the exact symbolic batch extent, carried through
        assert isinstance(valid[1], ir.ConstInt) and valid[1].value == 40
        assert isinstance(valid[2], ir.ConstInt) and valid[2].value == 96

    def test_batch_matmul_differing_rank_takes_batch_from_rank_present_operand(self):
        """Differing operand ranks: the batch valid extent is taken from whichever operand
        carries the batch dim — covers both the rhs-only and lhs-only broadcast arms.
        """
        # 2D lhs @ 3D rhs: batch from rhs.
        lhs2d = _mk_tile("lhs", [128, 128], [40, 64])  # 2D — no batch dim
        rhs3d = _mk_tile("rhs", [3, 128, 128], [3, 64, 96])  # 3D — batch = 3
        rt = tile.batch_matmul(lhs2d, rhs3d).type
        assert isinstance(rt, ir.TileType)
        assert _const_ints(rt.shape)[0] == 3  # output batch shape from rhs
        assert _valid_ints(rt) == [3, 40, 96]

        # 3D lhs @ 2D rhs: batch from lhs (the mirror arm).
        lhs3d = _mk_tile("lhs", [3, 128, 128], [3, 40, 64])  # 3D — batch = 3
        rhs2d = _mk_tile("rhs", [128, 128], [64, 96])  # 2D — no batch dim
        rt = tile.batch_matmul(lhs3d, rhs2d).type
        assert isinstance(rt, ir.TileType)
        assert _const_ints(rt.shape)[0] == 3
        assert _valid_ints(rt) == [3, 40, 96]

    """Tests for multi-dimensional TileType operations."""

    def test_transpose_3d(self):
        """Test transpose on 3D tile."""
        span = ir.Span.unknown()

        # Create a 3D tile [4, 8, 16]
        dim4 = ir.ConstInt(4, DataType.INT32, span)
        dim8 = ir.ConstInt(8, DataType.INT32, span)
        dim16 = ir.ConstInt(16, DataType.INT32, span)
        tile_type = ir.TileType([dim4, dim8, dim16], DataType.FP16)
        tile_var = ir.Var("tile", tile_type, span)

        # Transpose axes 0 and 2: [4, 8, 16] -> [16, 8, 4]
        call = tile.transpose(tile_var, 0, 2)

        assert isinstance(call, ir.Call)
        assert call.op.name == "tile.transpose"
        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        assert len(result_type.shape) == 3

    def test_row_max_3d(self):
        """Test row_max on 3D tile."""
        span = ir.Span.unknown()

        # Create a 3D tile [4, 16, 32]
        dim4 = ir.ConstInt(4, DataType.INT32, span)
        dim16 = ir.ConstInt(16, DataType.INT32, span)
        dim32 = ir.ConstInt(32, DataType.INT32, span)
        tile_type = ir.TileType([dim4, dim16, dim32], DataType.FP32)
        tile_var = ir.Var("tile", tile_type, span)
        tmp_tile = ir.Var("tmp_tile", tile_type, span)

        # row_max should reduce the last dimension: [4, 16, 32] -> [4, 16, 1]
        call = tile.row_max(tile_var, tmp_tile)

        assert isinstance(call, ir.Call)
        assert call.op.name == "tile.row_max"
        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        assert len(result_type.shape) == 3

    def test_slice_3d(self):
        """Test slice operation on 3D tile."""
        span = ir.Span.unknown()

        # Create a 3D tile [4, 16, 32]
        dim4 = ir.ConstInt(4, DataType.INT32, span)
        dim16 = ir.ConstInt(16, DataType.INT32, span)
        dim32 = ir.ConstInt(32, DataType.INT32, span)
        tile_type = ir.TileType([dim4, dim16, dim32], DataType.FP16)
        tile_var = ir.Var("tile", tile_type, span)

        # Create a slice with different shape [2, 8, 16]
        new_shape = [2, 8, 16]
        offset = [0, 0, 0]
        call = tile.slice(tile_var, new_shape, offset)

        assert isinstance(call, ir.Call)
        assert call.op.name == "tile.slice"
        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        assert len(result_type.shape) == 3


class TestTileBitwiseArithmeticOps:
    """Test suite for newly added tile-level bitwise and arithmetic ops (rem, and, or, xor)."""

    def test_tile_rem(self):
        """Test tile.rem operator - element-wise remainder of two tiles."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                b: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(a, [0, 0], [32, 32])
                tile_b: pl.Tile[[32, 32], pl.FP32] = pl.load(b, [0, 0], [32, 32])
                tmp: pl.Tile[[32, 32], pl.FP32] = pl.tile.create(
                    [32, 32], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
                )
                tile_c: pl.Tile[[32, 32], pl.FP32] = pl.rem(tile_a, tile_b, tmp)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.rem" in ir_str

    def test_tile_rems(self):
        """Test tile.rems operator - element-wise remainder of tile and scalar."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(a, [0, 0], [32, 32])
                tmp: pl.Tile[[32, 32], pl.FP32] = pl.tile.create(
                    [32, 32], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
                )
                tile_c: pl.Tile[[32, 32], pl.FP32] = pl.rems(tile_a, 3.0, tmp)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.rems" in ir_str

    @pytest.mark.parametrize("op_name", ["part_add", "part_mul", "part_max", "part_min"])
    def test_tile_part_ops(self, op_name):
        """Test tile.part_* partial-combine binary operators (tile-tile only)."""
        span = ir.Span.unknown()
        dim = ir.ConstInt(16, DataType.INT32, span)
        tile_type = ir.TileType([dim, dim], DataType.FP32)
        var_a = ir.Var("a", tile_type, span)
        var_b = ir.Var("b", tile_type, span)

        call = getattr(tile, op_name)(var_a, var_b)
        assert isinstance(call, ir.Call)
        assert call.op.name == f"tile.{op_name}"

    def test_tile_fmod(self):
        """Test tile.fmod operator - element-wise floating-point remainder of two tiles."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                b: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(a, [0, 0], [32, 32])
                tile_b: pl.Tile[[32, 32], pl.FP32] = pl.load(b, [0, 0], [32, 32])
                tile_c: pl.Tile[[32, 32], pl.FP32] = pl.fmod(tile_a, tile_b)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.fmod" in ir_str

    def test_tile_fmods(self):
        """Test tile.fmods operator - element-wise floating-point remainder of tile and scalar."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(a, [0, 0], [32, 32])
                tile_c: pl.Tile[[32, 32], pl.FP32] = pl.fmods(tile_a, 3.0)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.fmods" in ir_str

    def test_tile_and(self):
        """Test tile.and operator - element-wise bitwise AND of two tiles."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.INT32],
                b: pl.Tensor[[128, 128], pl.INT32],
                output: pl.Tensor[[128, 128], pl.INT32],
            ) -> pl.Tensor[[128, 128], pl.INT32]:
                tile_a: pl.Tile[[32, 32], pl.INT32] = pl.load(a, [0, 0], [32, 32])
                tile_b: pl.Tile[[32, 32], pl.INT32] = pl.load(b, [0, 0], [32, 32])
                tile_c: pl.Tile[[32, 32], pl.INT32] = pl.and_(tile_a, tile_b)
                result: pl.Tensor[[128, 128], pl.INT32] = pl.store(tile_c, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.and" in ir_str

    def test_tile_ands(self):
        """Test tile.ands operator - element-wise bitwise AND of tile and scalar."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.INT32],
                scalar: pl.Scalar[pl.INT32],
                output: pl.Tensor[[128, 128], pl.INT32],
            ) -> pl.Tensor[[128, 128], pl.INT32]:
                tile_a: pl.Tile[[32, 32], pl.INT32] = pl.load(a, [0, 0], [32, 32])
                tile_c: pl.Tile[[32, 32], pl.INT32] = pl.ands(tile_a, scalar)
                result: pl.Tensor[[128, 128], pl.INT32] = pl.store(tile_c, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.ands" in ir_str

    def test_tile_or(self):
        """Test tile.or operator - element-wise bitwise OR of two tiles."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.INT32],
                b: pl.Tensor[[128, 128], pl.INT32],
                output: pl.Tensor[[128, 128], pl.INT32],
            ) -> pl.Tensor[[128, 128], pl.INT32]:
                tile_a: pl.Tile[[32, 32], pl.INT32] = pl.load(a, [0, 0], [32, 32])
                tile_b: pl.Tile[[32, 32], pl.INT32] = pl.load(b, [0, 0], [32, 32])
                tile_c: pl.Tile[[32, 32], pl.INT32] = pl.or_(tile_a, tile_b)
                result: pl.Tensor[[128, 128], pl.INT32] = pl.store(tile_c, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.or" in ir_str

    def test_tile_ors(self):
        """Test tile.ors operator - element-wise bitwise OR of tile and scalar."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.INT32],
                scalar: pl.Scalar[pl.INT32],
                output: pl.Tensor[[128, 128], pl.INT32],
            ) -> pl.Tensor[[128, 128], pl.INT32]:
                tile_a: pl.Tile[[32, 32], pl.INT32] = pl.load(a, [0, 0], [32, 32])
                tile_c: pl.Tile[[32, 32], pl.INT32] = pl.ors(tile_a, scalar)
                result: pl.Tensor[[128, 128], pl.INT32] = pl.store(tile_c, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.ors" in ir_str

    def test_tile_xor(self):
        """Test tile.xor operator - element-wise bitwise XOR of two tiles with tmp buffer."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.INT32],
                b: pl.Tensor[[128, 128], pl.INT32],
                output: pl.Tensor[[128, 128], pl.INT32],
            ) -> pl.Tensor[[128, 128], pl.INT32]:
                tile_a: pl.Tile[[32, 32], pl.INT32] = pl.load(a, [0, 0], [32, 32])
                tile_b: pl.Tile[[32, 32], pl.INT32] = pl.load(b, [0, 0], [32, 32])
                tmp: pl.Tile[[32, 32], pl.INT32] = pl.tile.create(
                    [32, 32], dtype=pl.INT32, target_memory=pl.MemorySpace.Vec
                )
                tile_c: pl.Tile[[32, 32], pl.INT32] = pl.xor(tile_a, tile_b, tmp)
                result: pl.Tensor[[128, 128], pl.INT32] = pl.store(tile_c, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.xor" in ir_str

    def test_tile_xors(self):
        """Test tile.xors operator - element-wise bitwise XOR of tile and scalar with tmp buffer."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.INT32],
                scalar: pl.Scalar[pl.INT32],
                output: pl.Tensor[[128, 128], pl.INT32],
            ) -> pl.Tensor[[128, 128], pl.INT32]:
                tile_a: pl.Tile[[32, 32], pl.INT32] = pl.load(a, [0, 0], [32, 32])
                tmp: pl.Tile[[32, 32], pl.INT32] = pl.tile.create(
                    [32, 32], dtype=pl.INT32, target_memory=pl.MemorySpace.Vec
                )
                tile_c: pl.Tile[[32, 32], pl.INT32] = pl.xors(tile_a, scalar, tmp)
                result: pl.Tensor[[128, 128], pl.INT32] = pl.store(tile_c, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.xors" in ir_str

    def test_tile_shl(self):
        """Test tile.shl operator - element-wise bitwise left shift of two tiles."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.UINT32],
                b: pl.Tensor[[128, 128], pl.UINT32],
                output: pl.Tensor[[128, 128], pl.UINT32],
            ) -> pl.Tensor[[128, 128], pl.UINT32]:
                tile_a: pl.Tile[[16, 16], pl.UINT32] = pl.load(a, [0, 0], [16, 16])
                tile_b: pl.Tile[[16, 16], pl.UINT32] = pl.load(b, [0, 0], [16, 16])
                tile_c: pl.Tile[[16, 16], pl.UINT32] = pl.shl(tile_a, tile_b)
                result: pl.Tensor[[128, 128], pl.UINT32] = pl.store(tile_c, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.shl" in ir_str

    def test_tile_shls(self):
        """Test tile.shls operator - element-wise bitwise left shift of tile and scalar."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.UINT32],
                scalar: pl.Scalar[pl.INT32],
                output: pl.Tensor[[128, 128], pl.UINT32],
            ) -> pl.Tensor[[128, 128], pl.UINT32]:
                tile_a: pl.Tile[[16, 16], pl.UINT32] = pl.load(a, [0, 0], [16, 16])
                tile_c: pl.Tile[[16, 16], pl.UINT32] = pl.shls(tile_a, scalar)
                result: pl.Tensor[[128, 128], pl.UINT32] = pl.store(tile_c, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.shls" in ir_str

    def test_tile_maximums(self):
        """Test tile.maximums operator - element-wise maximum of tile and scalar."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[16, 16], pl.FP32] = pl.load(a, [0, 0], [16, 16])
                tile_c: pl.Tile[[16, 16], pl.FP32] = pl.maximums(tile_a, 0.0)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.maximums" in ir_str

    def test_tile_minimums(self):
        """Test tile.minimums operator - element-wise minimum of tile and scalar."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[16, 16], pl.FP32] = pl.load(a, [0, 0], [16, 16])
                tile_c: pl.Tile[[16, 16], pl.FP32] = pl.minimums(tile_a, 0.0)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.minimums" in ir_str

    def test_tile_shr(self):
        """Test tile.shr operator - element-wise bitwise right shift of two tiles."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.UINT32],
                b: pl.Tensor[[128, 128], pl.UINT32],
                output: pl.Tensor[[128, 128], pl.UINT32],
            ) -> pl.Tensor[[128, 128], pl.UINT32]:
                tile_a: pl.Tile[[16, 16], pl.UINT32] = pl.load(a, [0, 0], [16, 16])
                tile_b: pl.Tile[[16, 16], pl.UINT32] = pl.load(b, [0, 0], [16, 16])
                tile_c: pl.Tile[[16, 16], pl.UINT32] = pl.shr(tile_a, tile_b)
                result: pl.Tensor[[128, 128], pl.UINT32] = pl.store(tile_c, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.shr" in ir_str

    def test_tile_shrs(self):
        """Test tile.shrs operator - element-wise bitwise right shift of tile and scalar."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.UINT32],
                scalar: pl.Scalar[pl.INT32],
                output: pl.Tensor[[128, 128], pl.UINT32],
            ) -> pl.Tensor[[128, 128], pl.UINT32]:
                tile_a: pl.Tile[[16, 16], pl.UINT32] = pl.load(a, [0, 0], [16, 16])
                tile_c: pl.Tile[[16, 16], pl.UINT32] = pl.shrs(tile_a, scalar)
                result: pl.Tensor[[128, 128], pl.UINT32] = pl.store(tile_c, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.shrs" in ir_str

    def test_tile_shl_preserves_lhs_dtype(self):
        """Regression: tile.shl result dtype must match LHS dtype, not the promoted type.

        When lhs is UINT16 and rhs is UINT32, the result must be UINT16 (LHS dtype),
        consistent with the scalar variant tile.shls which preserves the LHS tile dtype.
        """

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.UINT16],
                b: pl.Tensor[[128, 128], pl.UINT32],
                output: pl.Tensor[[128, 128], pl.UINT16],
            ) -> pl.Tensor[[128, 128], pl.UINT16]:
                tile_a: pl.Tile[[16, 16], pl.UINT16] = pl.load(a, [0, 0], [16, 16])
                tile_b: pl.Tile[[16, 16], pl.UINT32] = pl.load(b, [0, 0], [16, 16])
                tile_c: pl.Tile[[16, 16], pl.UINT16] = pl.shl(tile_a, tile_b)
                result: pl.Tensor[[128, 128], pl.UINT16] = pl.store(tile_c, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.shl" in ir_str

    def test_tile_shr_preserves_lhs_dtype(self):
        """Regression: tile.shr result dtype must match LHS dtype, not the promoted type.

        When lhs is UINT16 and rhs is UINT32, the result must be UINT16 (LHS dtype),
        consistent with the scalar variant tile.shrs which preserves the LHS tile dtype.
        """

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.UINT16],
                b: pl.Tensor[[128, 128], pl.UINT32],
                output: pl.Tensor[[128, 128], pl.UINT16],
            ) -> pl.Tensor[[128, 128], pl.UINT16]:
                tile_a: pl.Tile[[16, 16], pl.UINT16] = pl.load(a, [0, 0], [16, 16])
                tile_b: pl.Tile[[16, 16], pl.UINT32] = pl.load(b, [0, 0], [16, 16])
                tile_c: pl.Tile[[16, 16], pl.UINT16] = pl.shr(tile_a, tile_b)
                result: pl.Tensor[[128, 128], pl.UINT16] = pl.store(tile_c, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.shr" in ir_str

    def test_tile_prelu(self):
        """Test tile.prelu operator - element-wise parametric ReLU with slope and tmp buffer."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_x: pl.Tile[[16, 16], pl.FP32] = pl.load(a, [0, 0], [16, 16])
                slope: pl.Tile[[16, 16], pl.FP32] = pl.tile.create(
                    [16, 16], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
                )
                tmp: pl.Tile[[16, 16], pl.FP32] = pl.tile.create(
                    [16, 16], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
                )
                tile_c: pl.Tile[[16, 16], pl.FP32] = pl.prelu(tile_x, slope, tmp)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.prelu" in ir_str

    def test_tile_not(self):
        """Test tile.not operator - element-wise bitwise NOT of a tile (int16/uint16 only)."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.INT16],
                output: pl.Tensor[[128, 128], pl.INT16],
            ) -> pl.Tensor[[128, 128], pl.INT16]:
                tile_a: pl.Tile[[16, 16], pl.INT16] = pl.load(a, [0, 0], [16, 16])
                tile_c: pl.Tile[[16, 16], pl.INT16] = pl.not_(tile_a)
                result: pl.Tensor[[128, 128], pl.INT16] = pl.store(tile_c, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.not" in ir_str

    def test_tile_addc(self):
        """Test tile.addc operator - element-wise addition of three tiles."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                b: pl.Tensor[[128, 128], pl.FP32],
                c: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(a, [0, 0], [32, 32])
                tile_b: pl.Tile[[32, 32], pl.FP32] = pl.load(b, [0, 0], [32, 32])
                tile_c: pl.Tile[[32, 32], pl.FP32] = pl.load(c, [0, 0], [32, 32])
                tile_out: pl.Tile[[32, 32], pl.FP32] = pl.addc(tile_a, tile_b, tile_c)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_out, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.addc" in ir_str

    def test_tile_subc(self):
        """Test tile.subc operator - element-wise subtraction of three tiles."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                b: pl.Tensor[[128, 128], pl.FP32],
                c: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(a, [0, 0], [32, 32])
                tile_b: pl.Tile[[32, 32], pl.FP32] = pl.load(b, [0, 0], [32, 32])
                tile_c: pl.Tile[[32, 32], pl.FP32] = pl.load(c, [0, 0], [32, 32])
                tile_out: pl.Tile[[32, 32], pl.FP32] = pl.subc(tile_a, tile_b, tile_c)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_out, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.subc" in ir_str

    def test_tile_addsc(self):
        """Test tile.addsc operator - element-wise addition of tile, scalar, and tile."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                b: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(a, [0, 0], [32, 32])
                tile_b: pl.Tile[[32, 32], pl.FP32] = pl.load(b, [0, 0], [32, 32])
                tile_out: pl.Tile[[32, 32], pl.FP32] = pl.addsc(tile_a, 2.0, tile_b)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_out, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.addsc" in ir_str

    def test_tile_subsc(self):
        """Test tile.subsc operator - element-wise subtraction of tile, scalar, and tile."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                b: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(a, [0, 0], [32, 32])
                tile_b: pl.Tile[[32, 32], pl.FP32] = pl.load(b, [0, 0], [32, 32])
                tile_out: pl.Tile[[32, 32], pl.FP32] = pl.subsc(tile_a, 2.0, tile_b)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_out, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.subsc" in ir_str

    def test_tile_lrelu(self):
        """Test tile.lrelu operator - element-wise leaky ReLU with scalar slope."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(a, [0, 0], [32, 32])
                tile_out: pl.Tile[[32, 32], pl.FP32] = pl.lrelu(tile_a, 0.1)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_out, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.lrelu" in ir_str

    def test_tile_sels(self):
        """Test tile.sels operator - select between two tiles via integer scalar mode."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                b: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(a, [0, 0], [32, 32])
                tile_b: pl.Tile[[32, 32], pl.FP32] = pl.load(b, [0, 0], [32, 32])
                tile_out: pl.Tile[[32, 32], pl.FP32] = pl.sels(tile_a, tile_b, 1)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_out, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.sels" in ir_str

    def test_tile_sel(self):
        """Test tile.sel operator - per-element selection between two tiles via mask tile."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                b: pl.Tensor[[128, 128], pl.FP32],
                m: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(a, [0, 0], [32, 32])
                tile_b: pl.Tile[[32, 32], pl.FP32] = pl.load(b, [0, 0], [32, 32])
                tile_m: pl.Tile[[32, 32], pl.FP32] = pl.load(m, [0, 0], [32, 32])
                tmp: pl.Tile[[1, 32], pl.UINT8] = pl.tile.create([1, 32], dtype=pl.UINT8)
                tile_out: pl.Tile[[32, 32], pl.FP32] = pl.sel(tile_m, tile_a, tile_b, tmp)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_out, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.sel" in ir_str


class TestTileLoadOp:
    """Tests for tile.load operation with valid_shapes and TileView."""

    def test_load_without_valid_shapes_sets_tileview_from_shapes(self):
        """When valid_shapes not provided, TileView.valid_shape equals shapes."""
        span = ir.Span.unknown()
        dim64 = ir.ConstInt(64, DataType.INT32, span)
        dim128 = ir.ConstInt(128, DataType.INT32, span)
        tensor_type = ir.TensorType([dim64, dim128], DataType.FP32)
        tensor = ir.Var("a", tensor_type, span)

        call = tile.load(tensor, [0, 0], [64, 128])
        tile_type = call.type

        assert isinstance(tile_type, ir.TileType)
        assert len(tile_type.get_effective_tile_view().valid_shape) == 2

    def test_load_with_static_valid_shapes_sets_tileview(self):
        """When valid_shapes provided as static ints, TileView.valid_shape reflects it."""
        span = ir.Span.unknown()
        dim64 = ir.ConstInt(64, DataType.INT32, span)
        dim128 = ir.ConstInt(128, DataType.INT32, span)
        tensor_type = ir.TensorType([dim64, dim128], DataType.FP32)
        tensor = ir.Var("a", tensor_type, span)

        call = tile.load(tensor, [0, 0], [128, 128], valid_shapes=[64, 128])
        tile_type = call.type

        assert isinstance(tile_type, ir.TileType)
        assert tile_type.tile_view is not None
        assert len(tile_type.tile_view.valid_shape) == 2
        # tile shape should still be [128, 128]
        assert len(tile_type.shape) == 2

    def test_load_with_dynamic_valid_shapes_sets_tileview(self):
        """When valid_shapes provided as symbolic vars, TileView.valid_shape uses them."""
        span = ir.Span.unknown()
        dim64 = ir.ConstInt(64, DataType.INT32, span)
        dim128 = ir.ConstInt(128, DataType.INT32, span)
        tensor_type = ir.TensorType([dim64, dim128], DataType.FP32)
        tensor = ir.Var("a", tensor_type, span)
        M = ir.Var("M", ir.ScalarType(DataType.INT64), span)
        N = ir.Var("N", ir.ScalarType(DataType.INT64), span)

        call = tile.load(tensor, [0, 0], [64, 128], valid_shapes=[M, N])
        tile_type = call.type

        assert isinstance(tile_type, ir.TileType)
        assert tile_type.tile_view is not None
        assert len(tile_type.tile_view.valid_shape) == 2
        # valid_shape elements should be the symbolic vars M and N
        assert tile_type.tile_view.valid_shape[0] is M
        assert tile_type.tile_view.valid_shape[1] is N

    def test_load_via_pl_load_with_valid_shapes(self):
        """pl.load with valid_shapes propagates TileView to the output tile."""

        @pl.program
        class Prog:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                M: pl.Scalar[pl.INT64],
                N: pl.Scalar[pl.INT64],
            ) -> pl.Tile[[128, 128], pl.FP32]:
                tile: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [0, 0], [128, 128], valid_shapes=[M, N])
                return tile

        # Just verifying it builds without error
        assert Prog is not None

    # ------------------------------------------------------------------
    # Q1 = INTERSECT: tile.load intersects the explicit valid_shapes= argument
    # with the source tensor's own valid region, never widening past it. Only
    # engaged when the source tensor carries an EXPLICIT valid region; a plain
    # tensor keeps valid_shapes verbatim (today's exact IR).
    # ------------------------------------------------------------------

    def _slice_with_valid(self, shape, valid_shape):
        """A tensor-typed slice Call whose result carries an explicit valid region."""
        span = ir.Span.unknown()
        base = ir.Var("a", ir.TensorType([128, 128], DataType.FP32), span)
        return tensor.slice(base, shape, [0, 0], valid_shape=valid_shape)

    def test_load_plain_tensor_keeps_valid_shapes_verbatim(self):
        """A plain source tensor (no explicit valid region) leaves valid_shapes untouched."""
        span = ir.Span.unknown()
        base = ir.Var("a", ir.TensorType([128, 128], DataType.FP32), span)
        m = ir.Var("M", ir.ScalarType(DataType.INDEX), span)
        n = ir.Var("N", ir.ScalarType(DataType.INDEX), span)

        call = tile.load(base, [0, 0], [64, 64], valid_shapes=[m, n])
        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        assert result_type.tile_view is not None
        valid_shape = result_type.tile_view.valid_shape

        # Verbatim: the exact symbolic vars, no intersect / Min injected.
        assert valid_shape[0] is m
        assert valid_shape[1] is n
        assert "min(" not in str(call).lower()

    def test_load_inherits_tensor_valid_region_when_omitted(self):
        """valid_shapes omitted -> tile inherits the source region (clipped to the window)."""
        span = ir.Span.unknown()
        vm = ir.Var("vm", ir.ScalarType(DataType.INDEX), span)
        sliced = self._slice_with_valid([64, 64], [vm, 64])
        src = ir.Var("s", sliced.type, span)

        call = tile.load(src, [0, 0], [64, 64])  # valid_shapes omitted -> defaults to shapes
        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        assert result_type.tile_view is not None
        valid_shape = result_type.tile_view.valid_shape

        # Row is the inherited vm capped at the tile height: min(vm, 64), not a bare 64.
        assert isinstance(valid_shape[0], ir.Min)
        assert valid_shape[0].left is vm
        assert isinstance(valid_shape[0].right, ir.ConstInt) and valid_shape[0].right.value == 64
        # Col: fully valid (64) and folds to a ConstInt.
        assert isinstance(valid_shape[1], ir.ConstInt) and valid_shape[1].value == 64

    def test_load_intersects_explicit_valid_shapes_with_tensor_region(self):
        """valid_shapes=<full tile shape> does NOT widen past the tensor's narrower region."""
        span = ir.Span.unknown()
        vm = ir.Var("vm", ir.ScalarType(DataType.INDEX), span)
        sliced = self._slice_with_valid([64, 64], [vm, 64])
        src = ir.Var("s", sliced.type, span)

        call = tile.load(src, [0, 0], [64, 64], valid_shapes=[64, 64])
        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        assert result_type.tile_view is not None
        valid_shape = result_type.tile_view.valid_shape

        # The whole point of INTERSECT: row is min(vm, 64), NOT the bare 64 argument.
        assert isinstance(valid_shape[0], ir.Min)
        assert valid_shape[0].left is vm
        assert not (isinstance(valid_shape[0], ir.ConstInt) and valid_shape[0].value == 64)
        assert isinstance(valid_shape[1], ir.ConstInt) and valid_shape[1].value == 64

    def test_load_clamps_valid_region_with_nonzero_offset(self):
        """A non-zero offset past the valid region clamps the inherited extent to 0."""
        span = ir.Span.unknown()
        # Source valid region is only 4 rows; the window starts at row 8, wholly
        # inside the padding, so no valid rows are visible: clamp(4 - 8, 0, 8) = 0.
        sliced = self._slice_with_valid([16, 16], [4, 16])
        src = ir.Var("s", sliced.type, span)

        call = tile.load(src, [8, 0], [8, 16])
        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        assert result_type.tile_view is not None
        valid_shape = result_type.tile_view.valid_shape

        assert isinstance(valid_shape[0], ir.ConstInt) and valid_shape[0].value == 0
        assert isinstance(valid_shape[1], ir.ConstInt) and valid_shape[1].value == 16

    def test_load_static_intersect_folds_to_constint(self):
        """A fully static intersect yields a ConstInt, never a Min node."""
        span = ir.Span.unknown()
        sliced = self._slice_with_valid([64, 64], [40, 64])
        src = ir.Var("s", sliced.type, span)

        call = tile.load(src, [0, 0], [64, 64])
        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        assert result_type.tile_view is not None
        valid_shape = result_type.tile_view.valid_shape

        # min(64, clamp(40 - 0, 0, 64)) = 40 — folded, no Min/Sub node.
        assert isinstance(valid_shape[0], ir.ConstInt) and valid_shape[0].value == 40
        assert isinstance(valid_shape[1], ir.ConstInt) and valid_shape[1].value == 64
        assert "min(" not in str(call).lower()


class TestTileCreateOp:
    """Tests for tile.create layout inference."""

    def test_create_column_vector_uses_col_major_layout(self):
        """Static `[N, 1]` Vec tiles should infer col-major block layout."""
        call = tile.create([32, 1], DataType.FP32, ir.MemorySpace.Vec)
        tile_type = call.type

        assert isinstance(tile_type, ir.TileType)
        eff = tile_type.get_effective_tile_view()
        assert eff.blayout == ir.TileLayout.col_major
        assert len(eff.valid_shape) == 2

    def test_create_row_vector_keeps_row_major_layout(self):
        """Non-column-vector shapes should keep the default row-major layout."""
        call = tile.create([1, 32], DataType.FP32, ir.MemorySpace.Vec)
        tile_type = call.type

        assert isinstance(tile_type, ir.TileType)
        assert tile_type.get_effective_tile_view().blayout == ir.TileLayout.row_major

    # ------------------------------------------------------------------
    # create_tile(valid_shape=...) — the accumulator surface. Under D2 an
    # unset valid_shape means "fully valid", so an empty-accumulator tile must
    # be creatable with valid_shape=[0, 0] for the assemble union rule to narrow.
    # ------------------------------------------------------------------

    def test_create_with_valid_shape_zero_accumulator(self):
        """create(valid_shape=[0, 0]) carries an explicit empty valid region."""
        call = tile.create([128, 128], DataType.FP32, valid_shape=[0, 0])
        tile_type = call.type

        assert isinstance(tile_type, ir.TileType)
        assert tile_type.tile_view is not None
        valid_shape = tile_type.tile_view.valid_shape
        assert len(valid_shape) == 2
        assert isinstance(valid_shape[0], ir.ConstInt) and valid_shape[0].value == 0
        assert isinstance(valid_shape[1], ir.ConstInt) and valid_shape[1].value == 0

    def test_create_with_partial_valid_shape_preserved(self):
        """A partial valid_shape (< shape) is preserved on the TileView."""
        call = tile.create([128, 128], DataType.FP32, valid_shape=[64, 64])
        tile_type = call.type

        assert isinstance(tile_type, ir.TileType)
        assert tile_type.tile_view is not None
        valid_shape = tile_type.tile_view.valid_shape
        assert isinstance(valid_shape[0], ir.ConstInt) and valid_shape[0].value == 64
        assert isinstance(valid_shape[1], ir.ConstInt) and valid_shape[1].value == 64

    def test_create_with_full_valid_shape_canonicalizes_to_no_view(self):
        """valid_shape == shape carries no information and collapses to no view (D2)."""
        call = tile.create([128, 128], DataType.FP32, valid_shape=[128, 128])
        tile_type = call.type

        assert isinstance(tile_type, ir.TileType)
        # Whole-view canonicalization: identical to an unset (fully valid) create.
        assert tile_type.tile_view is None
        plain = tile.create([128, 128], DataType.FP32)
        ir.assert_structural_equal(tile_type, plain.type)

    def test_create_rejects_valid_shape_exceeding_shape(self):
        """A ConstInt valid dim above the physical shape is rejected (never widen)."""
        with pytest.raises(ValueError, match=r"must be in \[0, shape dim"):
            tile.create([128, 128], DataType.FP32, valid_shape=[999, 1])

    def test_create_rejects_valid_shape_rank_mismatch(self):
        """A valid_shape whose rank differs from the shape is rejected."""
        with pytest.raises(ValueError, match="valid_shape rank"):
            tile.create([128, 128], DataType.FP32, valid_shape=[64])


class TestTileScalarOps:
    """Tests for tile scalar read/write ops (tile.read / tile.write)."""

    def test_tile_write_via_pl_write(self):
        """Test tile.write: write scalar into tile via pl.write with indices."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                src: pl.Tensor[[16, 16], pl.FP16],
                dst: pl.Tensor[[16, 16], pl.FP16],
            ) -> pl.Tensor[[16, 16], pl.FP16]:
                t: pl.Tile[[16, 16], pl.FP16] = pl.load(src, [0, 0], [16, 16])
                val: pl.Scalar[pl.FP16] = pl.read(t, [0, 0])
                pl.write(t, [0, 1], val)
                result: pl.Tensor[[16, 16], pl.FP16] = pl.store(t, [0, 0], dst)
                return result

        ir_str = str(Program)
        assert "tile.write" in ir_str

    def test_tile_read_write_direct(self):
        """Test tile.read/write via pl.tile.read/pl.tile.write directly."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                src: pl.Tensor[[16, 16], pl.FP16],
                dst: pl.Tensor[[16, 16], pl.FP16],
            ) -> pl.Tensor[[16, 16], pl.FP16]:
                t: pl.Tile[[16, 16], pl.FP16] = pl.load(src, [0, 0], [16, 16])
                val: pl.Scalar[pl.FP16] = pl.tile.read(t, [0, 0])
                pl.tile.write(t, [0, 1], val)
                result: pl.Tensor[[16, 16], pl.FP16] = pl.store(t, [0, 0], dst)
                return result

        ir_str = str(Program)
        assert "tile.read" in ir_str
        assert "tile.write" in ir_str


class TestTileAssembleOp:
    """Tests for tile.assemble operator."""

    def test_tile_assemble_basic(self):
        """Test tile.assemble type deduction returns target TileType."""
        span = ir.Span.unknown()

        dim16 = ir.ConstInt(16, DataType.INT32, span)
        dim128 = ir.ConstInt(128, DataType.INT32, span)
        dim64 = ir.ConstInt(64, DataType.INT32, span)

        target_type = ir.TileType([dim16, dim128], DataType.FP32)
        target_var = ir.Var("target", target_type, span)

        source_type = ir.TileType([dim16, dim64], DataType.FP32)
        source_var = ir.Var("source", source_type, span)

        call = tile.assemble(target_var, source_var, [0, 0])

        assert isinstance(call, ir.Call)
        assert call.op.name == "tile.assemble"
        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        assert result_type.dtype == DataType.FP32
        assert len(result_type.shape) == 2

    def test_tile_assemble_dtype_mismatch(self):
        """tile.assemble requires matching dtypes for target and source."""
        span = ir.Span.unknown()
        dim16 = ir.ConstInt(16, DataType.INT32, span)

        target_type = ir.TileType([dim16, dim16], DataType.FP32)
        target_var = ir.Var("target", target_type, span)

        source_type = ir.TileType([dim16, dim16], DataType.FP16)
        source_var = ir.Var("source", source_type, span)

        with pytest.raises(ValueError, match="same dtype"):
            tile.assemble(target_var, source_var, [0, 0])

    # ------------------------------------------------------------------
    # valid_shape union inference. assemble writes the source's
    # valid region [offset, offset + valid(source)) into the target; the result's
    # valid region is the UNION of that written rectangle with the target's existing
    # valid rectangle. A per-dim valid_shape describes ONE origin-anchored rectangle,
    # so the inference is the bounding box of the union ONLY when that union is itself
    # such a rectangle; a non-rectangular union (a gap, or an L-shape) is REJECTED via
    # CHECK_SPAN, never widened to the full physical shape.
    # ------------------------------------------------------------------

    @staticmethod
    def _valid_ints(result_type):
        """Return the result's valid_shape as ints; None if fully valid (no view)."""
        assert isinstance(result_type, ir.TileType)
        if result_type.tile_view is None or len(result_type.tile_view.valid_shape) == 0:
            return None
        out = []
        for e in result_type.tile_view.valid_shape:
            assert isinstance(e, ir.ConstInt), f"expected ConstInt valid dim, got {e}"
            out.append(e.value)
        return out

    def test_tile_assemble_accumulator_narrows(self):
        """assemble(create(valid_shape=[0,0]), src valid [32,64], [0,0]) -> [32,64]."""
        acc = tile.create([128, 128], DataType.FP32, valid_shape=[0, 0])
        src = tile.create([32, 64], DataType.FP32)
        call = tile.assemble(acc, src, [0, 0])
        assert self._valid_ints(call.type) == [32, 64]

    def test_tile_assemble_offset_into_empty_rejected(self):
        """The SAME source at offset [16,0] into an EMPTY accumulator is rejected.

        The written region is rows [16,48) x cols [0,64): it does not start at the
        origin, so the true valid set (== the written rectangle, since the target is
        empty) is not an origin-anchored rectangle and is unrepresentable as a per-dim
        valid_shape. The old bounding box max(0, 16+32)=48 widened rows [0,16) — never
        written — to valid; per the North Star that must reject, not widen. (Contrast
        ``test_tile_assemble_two_into_one_grows_monotonic``, where the same offset is
        <= the target's valid extent, so the union IS a contiguous rectangle.)
        """
        acc = tile.create([128, 128], DataType.FP32, valid_shape=[0, 0])
        src = tile.create([32, 64], DataType.FP32)
        with pytest.raises(ValueError, match="not representable"):
            tile.assemble(acc, src, [16, 0])

    def test_tile_assemble_non_rectangular_union_rejected(self):
        """A contiguous (no-gap) append whose union is an L-shape is rejected.

        acc valid grows to [32,128] after the first assemble; the second source is
        valid [64,16] at offset [32,0] — offset row 32 == acc valid row 32, so the
        rows abut with NO gap. Yet the union rows[0,32)xcols[0,128) U rows[32,96)x
        cols[0,16) is an L-shape (the corner rows[32,96)xcols[16,128) is written by
        neither operand). Its per-dim bounding box [96,128] would mark that corner
        valid, so the second assemble must reject even though the rows are contiguous.
        """
        acc = tile.create([128, 128], DataType.FP16, valid_shape=[0, 0])
        ta = tile.create([32, 128], DataType.FP16)
        tb = tile.create([64, 16], DataType.FP16)
        acc1 = tile.assemble(acc, ta, [0, 0])
        assert self._valid_ints(acc1.type) == [32, 128]
        with pytest.raises(ValueError, match="not representable"):
            tile.assemble(acc1, tb, [32, 0])

    def test_tile_assemble_multi_dim_grow_from_origin(self):
        """Overwriting a small valid region with a larger one at offset 0 grows both
        dims (containment): union == the larger source rectangle, still origin-anchored."""
        tgt = tile.create([128, 128], DataType.FP32, valid_shape=[16, 16])
        src = tile.create([64, 64], DataType.FP32)
        call = tile.assemble(tgt, src, [0, 0])
        assert self._valid_ints(call.type) == [64, 64]

    def test_tile_assemble_empty_source_is_noop(self):
        """A provably-empty source (valid_shape=[.,0]) writes nothing: the target's
        valid region is returned unchanged, never widened by the shifted offset."""
        acc = tile.create([128, 128], DataType.FP32, valid_shape=[0, 0])
        empty_src = tile.create([32, 64], DataType.FP32, valid_shape=[0, 0])
        call = tile.assemble(acc, empty_src, [16, 0])
        assert self._valid_ints(call.type) == [0, 0]

    def test_tile_assemble_two_into_one_grows_monotonic(self):
        """Two assembles into one accumulator grow the bounding box monotonically."""
        acc = tile.create([128, 128], DataType.FP32, valid_shape=[0, 0])
        src = tile.create([32, 64], DataType.FP32)
        first = tile.assemble(acc, src, [0, 0])
        assert self._valid_ints(first.type) == [32, 64]
        # Second assemble targets the first result (valid [32,64]) at offset [16,0]:
        # dim0 = max(32, 16+32) = 48, dim1 = max(64, 0+64) = 64. Box only grows.
        second = tile.assemble(first, src, [16, 0])
        assert self._valid_ints(second.type) == [48, 64]

    def test_tile_assemble_partial_target_absorbs_smaller_source(self):
        """A source fully inside the target's existing region leaves it unchanged."""
        tgt = tile.create([128, 128], DataType.FP32, valid_shape=[64, 64])
        src = tile.create([16, 16], DataType.FP32)
        call = tile.assemble(tgt, src, [0, 0])
        assert self._valid_ints(call.type) == [64, 64]

    def test_tile_assemble_dynamic_source_symbolic(self):
        """A dynamic source valid_shape yields a symbolic min/max box (compiles)."""
        span = ir.Span.unknown()
        vm = ir.Var("vm", ir.ScalarType(DataType.INDEX), span)
        vn = ir.Var("vn", ir.ScalarType(DataType.INDEX), span)
        acc = tile.create([128, 128], DataType.FP32, valid_shape=[0, 0])
        src_type = ir.TileType([128, 128], DataType.FP32, None, ir.TileView([vm, vn], [], None))
        src = ir.Var("src", src_type, span)
        call = tile.assemble(acc, src, [0, 0])

        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        assert result_type.tile_view is not None
        valid = result_type.tile_view.valid_shape
        assert len(valid) == 2
        # Each dim is min(shape, max(target_valid, offset + source_valid)) — symbolic,
        # not folded to a ConstInt.
        assert isinstance(valid[0], ir.Min) and not isinstance(valid[0], ir.ConstInt)
        assert isinstance(valid[1], ir.Min)

    def test_tile_assemble_out_of_bounds_rejected(self):
        """offset + valid(source) > shape (all ConstInt) is an out-of-bounds error."""
        tgt = tile.create([128, 128], DataType.FP32)
        src = tile.create([64, 64], DataType.FP32)
        with pytest.raises(ValueError, match="out-of-bounds assemble"):
            tile.assemble(tgt, src, [80, 0])  # 80 + 64 = 144 > 128

    def test_tile_assemble_full_into_full_canonicalizes(self):
        """A fully-valid source into a fully-valid target stays fully valid (no view).

        This guards D2 canonicalization (a whole-view valid_shape collapses to a
        nullopt tile_view), not the 4b union: a fully-valid tile target is nullopt-view
        by construction, so the ``tile_view.has_value()`` guard is false in both the
        old and new code. The out-of-bounds rejection for a viewless target is covered
        by ``test_tile_assemble_out_of_bounds_rejected``.
        """
        tgt = tile.create([128, 128], DataType.FP32)
        src = tile.create([128, 128], DataType.FP32)
        call = tile.assemble(tgt, src, [0, 0])
        # union == full shape -> canonicalizes to no view.
        assert self._valid_ints(call.type) is None
        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        assert result_type.tile_view is None

    # ------------------------------------------------------------------
    # Adversary regression: the representability proof is PER-DIM. A symbolic dim
    # (a passenger, or a masking row) must NOT disable the rejection of a provable
    # widening in the remaining static dims — the old global "all dims static" gate
    # deferred the whole check to the widening box whenever any dim was dynamic.
    # ------------------------------------------------------------------

    def test_tile_assemble_symbolic_row_static_col_grow_rejected(self):
        """target valid [m,32], source valid [4,128] at origin: dim1 provably grows
        cols 32->128 while dim0's coverage (4 vs symbolic m) is unprovable, so for the
        common runtime case m>4 the union is an L-shape. The old global gate saw m
        symbolic and widened cols [32,128) to valid for rows the source never wrote
        there — the North Star forbids it. Rejection must fire despite the symbolic m.
        """
        span = ir.Span.unknown()
        m = ir.Var("m", ir.ScalarType(DataType.INDEX), span)
        tgt = _mk_tile("tgt", [128, 128], [m, 32])
        src = _mk_tile("src", [128, 128], [4, 128])
        with pytest.raises(ValueError, match="not representable"):
            tile.assemble(tgt, src, [0, 0])

    def test_tile_assemble_symbolic_passenger_masks_static_lshape_rejected(self):
        """3D adversary: a symbolic PASSENGER dim (dim2, equal on both operands) must
        not disable rejection of a provably static L-shape in dims 0/1. target
        [4,2,d2], source [2,4,d2] at origin form an L-shape in the (dim0,dim1) plane
        for every d2; the old gate saw dim2 symbolic and widened dims 0/1 to the [4,4]
        bounding box, marking cells written by neither operand as valid.
        """
        span = ir.Span.unknown()
        d2 = ir.Var("d2", ir.ScalarType(DataType.INDEX), span)
        tgt = _mk_tile("t", [8, 8, 8], [4, 2, d2])
        src = _mk_tile("s", [8, 8, 8], [2, 4, d2])
        with pytest.raises(ValueError, match="not representable"):
            tile.assemble(tgt, src, [0, 0, 0])

    def test_tile_assemble_static_lshape_all_const_rejected(self):
        """The fully-static analog of the passenger-masked L-shape also rejects — the
        symbolic passenger dim is not what makes it non-representable."""
        tgt = _mk_tile("t", [8, 8, 8], [4, 2, 8])
        src = _mk_tile("s", [8, 8, 8], [2, 4, 8])
        with pytest.raises(ValueError, match="not representable"):
            tile.assemble(tgt, src, [0, 0, 0])

    def test_tile_assemble_symbolic_contiguous_append_accepted(self):
        """A SYMBOLIC contiguous append is provable and must be accepted.

        ``assemble(acc /*valid [v, 128]*/, src /*valid [32, 128]*/, offset=[v, 0])``
        writes exactly at the target's current valid boundary, so the union stays an
        origin-anchored rectangle even though neither extent is a ConstInt. The
        ``no_gap`` predicate discharges it via structural equality of ``offset[0]``
        and ``target_valid[0]`` — both are the same ``v`` node. Before that arm
        existed the accumulator idiom was rejected merely for being dynamic.
        """
        span = ir.Span.unknown()
        v = ir.Var("v", ir.ScalarType(DataType.INDEX), span)
        tgt = _mk_tile("t", [128, 128], [v, 128])
        src = _mk_tile("s", [128, 128], [32, 128])
        call = tile.assemble(tgt, src, [v, 0])
        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        assert result_type.tile_view is not None
        out_valid = result_type.tile_view.valid_shape
        # Row extent grows to max(v, v + 32) clamped to the physical 128; not widened
        # to the full shape, and not collapsed to the source's 32.
        assert not isinstance(out_valid[0], ir.ConstInt), "row extent must stay symbolic"
        assert isinstance(out_valid[1], ir.ConstInt) and out_valid[1].value == 128

    def test_tile_assemble_symbolic_offset_into_empty_target_rejected(self):
        """A symbolic offset into an EMPTY target leaves a provable gap [0, off)."""
        span = ir.Span.unknown()
        v = ir.Var("v", ir.ScalarType(DataType.INDEX), span)
        tgt = _mk_tile("t", [128, 128], [0, 128])
        src = _mk_tile("s", [128, 128], [32, 128])
        with pytest.raises(ValueError, match="not representable"):
            tile.assemble(tgt, src, [v, 0])

    def test_tile_assemble_distinct_symbolic_offset_rejected(self):
        """Two DIFFERENT symbolic extents are not provably equal, so the append cannot
        be shown gap-free. The structural-equality arm must not over-accept."""
        span = ir.Span.unknown()
        v = ir.Var("v", ir.ScalarType(DataType.INDEX), span)
        w = ir.Var("w", ir.ScalarType(DataType.INDEX), span)
        tgt = _mk_tile("t", [128, 128], [v, 128])
        src = _mk_tile("s", [128, 128], [32, 128])
        with pytest.raises(ValueError, match="not representable"):
            tile.assemble(tgt, src, [w, 0])

    def test_tile_assemble_static_contiguous_append_accepted(self):
        """Regression guard: the all-static contiguous append still folds to [64, 128]."""
        tgt = _mk_tile("t", [128, 128], [32, 128])
        src = _mk_tile("s", [128, 128], [32, 128])
        call = tile.assemble(tgt, src, [32, 0])
        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        assert result_type.tile_view is not None
        out_valid = result_type.tile_view.valid_shape
        assert isinstance(out_valid[0], ir.ConstInt) and out_valid[0].value == 64
        assert isinstance(out_valid[1], ir.ConstInt) and out_valid[1].value == 128

    def test_tile_assemble_dynamic_source_into_full_target_not_rejected(self):
        """A symbolic source assembled into a FULLY-VALID target must NOT reject: the
        box clamps to the physical shape, so the union is provably the full target
        (Form target-full). Guards against over-rejecting the common
        dynamic-source-into-full-accumulator pattern; the result stays fully valid.
        """
        span = ir.Span.unknown()
        vm = ir.Var("vm", ir.ScalarType(DataType.INDEX), span)
        vn = ir.Var("vn", ir.ScalarType(DataType.INDEX), span)
        tgt = _mk_tile("tgt", [128, 128])  # fully valid (no explicit view)
        src_type = ir.TileType([128, 128], DataType.FP32, None, ir.TileView([vm, vn], [], None))
        src = ir.Var("src", src_type, span)
        call = tile.assemble(tgt, src, [0, 0])  # must not raise
        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        assert result_type.tile_view is None  # full target -> full (viewless) result


class TestTileExtractOp:
    """Tests for tile.extract operator (ISA TEXTRACT Variant 1)."""

    @staticmethod
    def _make_src_var(
        rows: int = 64,
        cols: int = 256,
        dtype: DataType = DataType.FP16,
        memory_space: ir.MemorySpace | None = None,
    ) -> ir.Var:
        span = ir.Span.unknown()
        r = ir.ConstInt(rows, DataType.INT32, span)
        c = ir.ConstInt(cols, DataType.INT32, span)
        tile_type = ir.TileType([r, c], dtype, memory_space=memory_space)
        return ir.Var("src", tile_type, span)

    def test_tile_extract_basic(self):
        """tile.extract returns a TileType with the requested shape and src dtype."""
        src_var = self._make_src_var()

        call = tile.extract(src_var, 0, 0, shape=[64, 64], target_memory=ir.MemorySpace.Left)

        assert isinstance(call, ir.Call)
        assert call.op.name == "tile.extract"
        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        assert result_type.dtype == DataType.FP16
        assert len(result_type.shape) == 2
        rows, cols = result_type.shape
        assert isinstance(rows, ir.ConstInt) and rows.value == 64
        assert isinstance(cols, ir.ConstInt) and cols.value == 64

    def test_tile_extract_acc_to_mat(self):
        """Acc source → Mat target: src lives in Acc, dtype preserved."""
        src_var = self._make_src_var(64, 64, DataType.FP32, memory_space=ir.MemorySpace.Acc)
        src_tile_type = src_var.type
        assert isinstance(src_tile_type, ir.TileType)
        assert src_tile_type.memory_space == ir.MemorySpace.Acc

        call = tile.extract(src_var, 0, 0, shape=[32, 32], target_memory=ir.MemorySpace.Mat)

        assert call.op.name == "tile.extract"
        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        assert result_type.dtype == DataType.FP32
        rows, cols = result_type.shape
        assert isinstance(rows, ir.ConstInt) and rows.value == 32
        assert isinstance(cols, ir.ConstInt) and cols.value == 32

    def test_tile_extract_dynamic_offset(self):
        """Runtime symbolic offsets are accepted (no compile-time bounds check fires)."""
        span = ir.Span.unknown()
        src_var = self._make_src_var()
        row = ir.Var("row", ir.ScalarType(DataType.INDEX), span)
        col = ir.Var("col", ir.ScalarType(DataType.INDEX), span)

        call = tile.extract(src_var, row, col, shape=[16, 16], target_memory=ir.MemorySpace.Left)

        assert call.op.name == "tile.extract"
        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        rows, cols = result_type.shape
        assert isinstance(rows, ir.ConstInt) and rows.value == 16
        assert isinstance(cols, ir.ConstInt) and cols.value == 16

    # ------------------------------------------------------------------
    # valid_shape propagation: extract CLIPS the window to the
    # source's own valid region. A fully-valid source (bounds-checked in-window)
    # yields a fully-valid dst; only an explicit partial source narrows the result.
    # ------------------------------------------------------------------

    @staticmethod
    def _partial_src(rows, cols, valid, dtype=DataType.FP16):
        span = ir.Span.unknown()
        shape = [ir.ConstInt(rows, DataType.INT32, span), ir.ConstInt(cols, DataType.INT32, span)]
        vs = [ir.ConstInt(v, DataType.INDEX, span) if isinstance(v, int) else v for v in valid]
        view = ir.TileView(valid_shape=vs)
        return ir.Var("src", ir.TileType(shape, dtype, None, view), span)

    def test_tile_extract_fully_valid_source_unchanged(self):
        """A fully-valid source extracts a fully-valid window (dst_shape)."""
        result_type = tile.extract(
            self._make_src_var(64, 256), 0, 0, shape=[32, 32], target_memory=ir.MemorySpace.Left
        ).type
        assert isinstance(result_type, ir.TileType)
        vs = result_type.get_effective_tile_view().valid_shape
        assert _const_ints(vs) == [32, 32]

    def test_tile_extract_clips_partial_source(self):
        """Partial src valid [40, 50], extract [32, 32] @ [20, 40] ->
        clamp(40 - 20, 0, 32) = 20, clamp(50 - 40, 0, 32) = 10."""
        result_type = tile.extract(
            self._partial_src(64, 256, [40, 50]), 20, 40, shape=[32, 32], target_memory=ir.MemorySpace.Left
        ).type
        assert isinstance(result_type, ir.TileType)
        vs = result_type.get_effective_tile_view().valid_shape
        assert _const_ints(vs) == [20, 10]

    def test_tile_extract_partial_source_window_fully_inside(self):
        """When the window lies inside the valid region the whole dst is valid."""
        result_type = tile.extract(
            self._partial_src(64, 256, [40, 50]), 0, 0, shape=[32, 32], target_memory=ir.MemorySpace.Left
        ).type
        assert isinstance(result_type, ir.TileType)
        vs = result_type.get_effective_tile_view().valid_shape
        assert _const_ints(vs) == [32, 32]

    def test_tile_extract_dynamic_offset_symbolic(self):
        """A dynamic row offset into a partial source yields a symbolic (min/max) extent."""
        span = ir.Span.unknown()
        r = ir.Var("r", ir.ScalarType(DataType.INDEX), span)
        result_type = tile.extract(
            self._partial_src(64, 256, [40, 50]), r, 0, shape=[32, 32], target_memory=ir.MemorySpace.Left
        ).type
        assert isinstance(result_type, ir.TileType)
        vs = result_type.get_effective_tile_view().valid_shape
        assert isinstance(vs[0], ir.Min) and not isinstance(vs[0], ir.ConstInt)
        assert isinstance(vs[1], ir.ConstInt) and vs[1].value == 32

    def test_tile_extract_shape_exceeds_src_static(self):
        """Static shape larger than src is rejected at deduction time."""
        src_var = self._make_src_var(64, 64)

        with pytest.raises(ValueError, match="exceeds src"):
            tile.extract(src_var, 0, 0, shape=[128, 128], target_memory=ir.MemorySpace.Left)

    def test_tile_extract_offset_plus_shape_exceeds_src_static(self):
        """Constant offset + shape that walks past src is rejected at deduction."""
        src_var = self._make_src_var(64, 64)

        # offset 60 + shape 16 = 76 > 64 rows
        with pytest.raises(ValueError, match="exceeds src row"):
            tile.extract(src_var, 60, 0, shape=[16, 16], target_memory=ir.MemorySpace.Left)

    def test_tile_extract_negative_offset_static(self):
        """Constant negative offset is rejected at deduction."""
        src_var = self._make_src_var(64, 64)

        with pytest.raises(ValueError, match="must be >= 0"):
            tile.extract(src_var, -1, 0, shape=[16, 16], target_memory=ir.MemorySpace.Left)

    def test_tile_extract_rejects_non_index_offset(self):
        """index_row/col must be INT64/UINT64/INDEX."""
        span = ir.Span.unknown()
        src_var = self._make_src_var()
        bad = ir.Var("bad", ir.ScalarType(DataType.FP32), span)

        with pytest.raises(ValueError, match="INT64/UINT64/INDEX"):
            tile.extract(src_var, bad, 0, shape=[16, 16], target_memory=ir.MemorySpace.Left)

    def test_tile_extract_rejects_dynamic_shape(self):
        """shape elements must be compile-time ConstInt for storage allocation."""
        span = ir.Span.unknown()
        src_var = self._make_src_var()
        dyn = ir.Var("dyn", ir.ScalarType(DataType.INDEX), span)

        with pytest.raises(ValueError, match="compile-time ConstInt"):
            tile.extract(src_var, 0, 0, shape=[dyn, 16], target_memory=ir.MemorySpace.Left)

    def test_tile_extract_rejects_non_2d_shape(self):
        """shape must be 2D."""
        src_var = self._make_src_var()

        with pytest.raises(ValueError, match="2D"):
            tile.extract(src_var, 0, 0, shape=[16, 16, 16], target_memory=ir.MemorySpace.Left)


class TestTileScatterUpdateOps:
    """Test suite for tile.scatter_update operation."""

    @pytest.mark.parametrize(
        ("input_shape", "src_shape", "dtype"),
        [
            # 2D scatter: rows=16, src first dim = b*s = 8.
            ([16, 64], [8, 64], DataType.FP16),
            # 4D KV-cache style: [block_num, block_size, 1, d] with src [b, s, 1, d].
            ([4, 4, 1, 64], [2, 4, 1, 64], DataType.BF16),
        ],
        ids=["2d", "4d"],
    )
    def test_tile_scatter_update_valid(self, input_shape, src_shape, dtype):
        """tile.scatter_update preserves input rank/dtype across 2D and 4D inputs."""
        span = ir.Span.unknown()
        input_type = ir.TileType(_const_dims(span, *input_shape), dtype)
        index_type = ir.TileType(_const_dims(span, 2, 4), DataType.INT32)
        src_type = ir.TileType(_const_dims(span, *src_shape), dtype)

        call = tile.scatter_update(
            ir.Var("inp", input_type, span),
            -2,
            ir.Var("idx", index_type, span),
            ir.Var("src", src_type, span),
        )

        assert isinstance(call, ir.Call)
        assert call.op.name == "tile.scatter_update"
        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        assert result_type.dtype == dtype
        const_dims = [dim for dim in result_type.shape if isinstance(dim, ir.ConstInt)]
        assert len(const_dims) == len(result_type.shape)
        assert [dim.value for dim in const_dims] == input_shape

    def test_tile_scatter_update_keeps_implicit_column_vector_layout(self):
        """Same alias rule as tile.scatter: a `[M, 1]` input is implicitly col_major,
        so the result must stay implicit rather than pin the raw TileView defaults."""
        span = ir.Span.unknown()
        colvec = ir.TileType(_const_dims(span, 64, 1), DataType.FP32)
        idx_type = ir.TileType(_const_dims(span, 64, 1), DataType.INT32)
        assert colvec.tile_view is None, "input leaves the view implicit"

        result_type = tile.scatter_update(
            ir.Var("inp", colvec, span),
            -2,
            ir.Var("idx", idx_type, span),
            ir.Var("src", colvec, span),
        ).type

        assert isinstance(result_type, ir.TileType)
        assert result_type.tile_view is None

    @pytest.mark.parametrize(
        ("src_dtype", "dim", "match"),
        [
            (DataType.FP32, -2, "src dtype"),  # input is FP16; src must match
            (DataType.FP16, -1, "dim=-2"),  # only dim=-2 is supported
        ],
        ids=["dtype_mismatch", "invalid_dim"],
    )
    def test_tile_scatter_update_rejects_invalid(self, src_dtype, dim, match):
        """tile.scatter_update validates src dtype and the dim argument."""
        span = ir.Span.unknown()
        input_type = ir.TileType(_const_dims(span, 16, 64), DataType.FP16)
        index_type = ir.TileType(_const_dims(span, 2, 4), DataType.INT32)
        src_type = ir.TileType(_const_dims(span, 8, 64), src_dtype)

        with pytest.raises(ValueError, match=match):
            tile.scatter_update(
                ir.Var("inp", input_type, span),
                dim,
                ir.Var("idx", index_type, span),
                ir.Var("src", src_type, span),
            )


class TestTileMscatterOps:
    """Test suite for tile.mscatter operation."""

    def test_tile_mscatter_basic(self):
        """Test tile.mscatter constructs a Call returning a TensorType."""
        span = ir.Span.unknown()
        rows = ir.ConstInt(16, DataType.INT32, span)
        cols = ir.ConstInt(32, DataType.INT32, span)
        tensor_n = ir.ConstInt(1024, DataType.INT32, span)

        src_type = ir.TileType([rows, cols], DataType.FP32)
        idx_type = ir.TileType([rows, cols], DataType.INT32)
        tensor_type = ir.TensorType([tensor_n], DataType.FP32)

        src_var = ir.Var("src", src_type, span)
        idx_var = ir.Var("idx", idx_type, span)
        out_var = ir.Var("out", tensor_type, span)

        call = tile.mscatter(src_var, idx_var, out_var)

        assert isinstance(call, ir.Call)
        assert call.op.name == "tile.mscatter"
        result_type = call.type
        assert isinstance(result_type, ir.TensorType)
        assert result_type.dtype == DataType.FP32

    def test_tile_mscatter_fp16(self):
        """Test tile.mscatter works with FP16 dtype."""
        span = ir.Span.unknown()
        rows = ir.ConstInt(8, DataType.INT32, span)
        cols = ir.ConstInt(16, DataType.INT32, span)
        tensor_n = ir.ConstInt(512, DataType.INT32, span)

        src_type = ir.TileType([rows, cols], DataType.FP16)
        idx_type = ir.TileType([rows, cols], DataType.INT32)
        tensor_type = ir.TensorType([tensor_n], DataType.FP16)

        src_var = ir.Var("src", src_type, span)
        idx_var = ir.Var("idx", idx_type, span)
        out_var = ir.Var("out", tensor_type, span)

        call = tile.mscatter(src_var, idx_var, out_var)
        assert call.op.name == "tile.mscatter"
        result_type = call.type
        assert isinstance(result_type, ir.TensorType)
        assert result_type.dtype == DataType.FP16

    def test_tile_mscatter_src_dtype_error(self):
        """Test tile.mscatter rejects unsupported src dtype."""
        span = ir.Span.unknown()
        rows = ir.ConstInt(16, DataType.INT32, span)
        cols = ir.ConstInt(32, DataType.INT32, span)
        tensor_n = ir.ConstInt(1024, DataType.INT32, span)

        src_type = ir.TileType([rows, cols], DataType.UINT8)  # unsupported
        idx_type = ir.TileType([rows, cols], DataType.INT32)
        tensor_type = ir.TensorType([tensor_n], DataType.UINT8)

        src_var = ir.Var("src", src_type, span)
        idx_var = ir.Var("idx", idx_type, span)
        out_var = ir.Var("out", tensor_type, span)

        with pytest.raises(ValueError, match="src dtype"):
            tile.mscatter(src_var, idx_var, out_var)

    def test_tile_mscatter_idx_dtype_error(self):
        """Test tile.mscatter rejects non-INT32 idx dtype."""
        span = ir.Span.unknown()
        rows = ir.ConstInt(16, DataType.INT32, span)
        cols = ir.ConstInt(32, DataType.INT32, span)
        tensor_n = ir.ConstInt(1024, DataType.INT32, span)

        src_type = ir.TileType([rows, cols], DataType.FP32)
        idx_type = ir.TileType([rows, cols], DataType.INT16)  # wrong dtype
        tensor_type = ir.TensorType([tensor_n], DataType.FP32)

        src_var = ir.Var("src", src_type, span)
        idx_var = ir.Var("idx", idx_type, span)
        out_var = ir.Var("out", tensor_type, span)

        with pytest.raises(ValueError, match="idx dtype"):
            tile.mscatter(src_var, idx_var, out_var)

    def test_tile_mscatter_rank_mismatch_error(self):
        """Test tile.mscatter rejects idx with different rank than src."""
        span = ir.Span.unknown()
        rows = ir.ConstInt(16, DataType.INT32, span)
        cols = ir.ConstInt(32, DataType.INT32, span)
        tensor_n = ir.ConstInt(1024, DataType.INT32, span)

        src_type = ir.TileType([rows, cols], DataType.FP32)  # 2D
        idx_type = ir.TileType([rows], DataType.INT32)  # 1D
        tensor_type = ir.TensorType([tensor_n], DataType.FP32)

        src_var = ir.Var("src", src_type, span)
        idx_var = ir.Var("idx", idx_type, span)
        out_var = ir.Var("out", tensor_type, span)

        with pytest.raises(ValueError, match="idx rank"):
            tile.mscatter(src_var, idx_var, out_var)

    def test_tile_mscatter_dtype_mismatch_error(self):
        """Test tile.mscatter rejects output_tensor with dtype different from src."""
        span = ir.Span.unknown()
        rows = ir.ConstInt(16, DataType.INT32, span)
        cols = ir.ConstInt(32, DataType.INT32, span)
        tensor_n = ir.ConstInt(1024, DataType.INT32, span)

        src_type = ir.TileType([rows, cols], DataType.FP32)
        idx_type = ir.TileType([rows, cols], DataType.INT32)
        tensor_type = ir.TensorType([tensor_n], DataType.FP16)  # mismatched

        src_var = ir.Var("src", src_type, span)
        idx_var = ir.Var("idx", idx_type, span)
        out_var = ir.Var("out", tensor_type, span)

        with pytest.raises(ValueError, match="output_tensor dtype"):
            tile.mscatter(src_var, idx_var, out_var)

    def test_tile_mscatter_arg_count_error(self):
        """Test tile.mscatter rejects wrong number of arguments."""
        span = ir.Span.unknown()
        rows = ir.ConstInt(16, DataType.INT32, span)
        cols = ir.ConstInt(32, DataType.INT32, span)

        src_type = ir.TileType([rows, cols], DataType.FP32)
        idx_type = ir.TileType([rows, cols], DataType.INT32)

        src_var = ir.Var("src", src_type, span)
        idx_var = ir.Var("idx", idx_type, span)

        with pytest.raises(ValueError, match="3 arguments"):
            # Missing output_tensor; call the op directly via create_op_call
            ir.create_op_call("tile.mscatter", [src_var, idx_var], {}, span)

    def test_tile_mscatter_shape_mismatch_error(self):
        """Test tile.mscatter rejects idx with different shape than src."""
        span = ir.Span.unknown()
        src_type = ir.TileType(
            [ir.ConstInt(16, DataType.INT32, span), ir.ConstInt(32, DataType.INT32, span)],
            DataType.FP32,
        )
        idx_type = ir.TileType(
            [ir.ConstInt(16, DataType.INT32, span), ir.ConstInt(64, DataType.INT32, span)],
            DataType.INT32,
        )
        tensor_type = ir.TensorType([ir.ConstInt(1024, DataType.INT32, span)], DataType.FP32)

        src_var = ir.Var("src", src_type, span)
        idx_var = ir.Var("idx", idx_type, span)
        out_var = ir.Var("out", tensor_type, span)

        with pytest.raises(ValueError, match="idx shape to match src shape"):
            tile.mscatter(src_var, idx_var, out_var)

    def test_tile_mscatter_scalar_output_error(self):
        """Test tile.mscatter rejects scalar (rank-0) output tensor."""
        span = ir.Span.unknown()
        rows = ir.ConstInt(16, DataType.INT32, span)
        cols = ir.ConstInt(32, DataType.INT32, span)

        src_type = ir.TileType([rows, cols], DataType.FP32)
        idx_type = ir.TileType([rows, cols], DataType.INT32)
        tensor_type = ir.TensorType([], DataType.FP32)

        src_var = ir.Var("src", src_type, span)
        idx_var = ir.Var("idx", idx_type, span)
        out_var = ir.Var("out", tensor_type, span)

        with pytest.raises(ValueError, match="at least 1 dimension"):
            tile.mscatter(src_var, idx_var, out_var)


class TestTileScatterOps:
    """Test suite for tile.scatter (index form, DPS)."""

    @pytest.mark.parametrize(
        ("dtype", "idx_dtype"),
        [
            (DataType.FP32, DataType.INT32),
            (DataType.INT32, DataType.INT32),
            (DataType.FP16, DataType.INT16),
            (DataType.BF16, DataType.INT16),
            (DataType.INT16, DataType.INT16),
            (DataType.INT8, DataType.INT16),
        ],
        ids=["fp32-i32", "i32-i32", "fp16-i16", "bf16-i16", "i16-i16", "i8-i16"],
    )
    def test_tile_scatter_valid(self, dtype, idx_dtype):
        """tile.scatter constructs a Call returning a TileType aliased to dst."""
        span = ir.Span.unknown()
        src_type = ir.TileType(_const_dims(span, 4, 32), dtype)
        # indexes are per-element flattened indices, same shape as src.
        idx_type = ir.TileType(_const_dims(span, 4, 32), idx_dtype)
        dst_type = ir.TileType(_const_dims(span, 16, 32), dtype)

        call = tile.scatter(
            ir.Var("dst", dst_type, span),
            ir.Var("src", src_type, span),
            ir.Var("idx", idx_type, span),
        )

        assert isinstance(call, ir.Call)
        assert call.op.name == "tile.scatter"
        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        assert result_type.dtype == dtype
        const_dims = [dim.value for dim in result_type.shape if isinstance(dim, ir.ConstInt)]
        assert const_dims == [16, 32]

    def test_tile_scatter_keeps_implicit_column_vector_layout(self):
        """The result aliases `dst`, so it must not pin the raw TileView defaults.

        A `[M, 1]` tile that leaves `tile_view` implicit is col_major (see
        `InferImplicitTileLayoutFromShape`). Seeding the alias's TileView from a
        default-constructed one would stamp an explicit row_major / none_box /
        fractal=512 view onto a buffer whose own `pto.alloc_tile` declares
        col_major. Staying implicit (`tile_view is None`) is the canonical form:
        `TileType` collapses a view equal to the implicit one back to None.
        """
        span = ir.Span.unknown()
        colvec = ir.TileType(_const_dims(span, 64, 1), DataType.FP32)
        idx_type = ir.TileType(_const_dims(span, 64, 1), DataType.INT32)
        assert colvec.tile_view is None, "source leaves the view implicit"

        result_type = tile.scatter(
            ir.Var("dst", colvec, span),
            ir.Var("src", colvec, span),
            ir.Var("idx", idx_type, span),
        ).type

        assert isinstance(result_type, ir.TileType)
        assert result_type.tile_view is None

    def test_tile_scatter_rejects_dtype_mismatch(self):
        """tile.scatter requires dst dtype to match src dtype."""
        span = ir.Span.unknown()
        src_type = ir.TileType(_const_dims(span, 4, 32), DataType.FP32)
        idx_type = ir.TileType(_const_dims(span, 4, 1), DataType.INT32)
        dst_type = ir.TileType(_const_dims(span, 16, 32), DataType.FP16)

        with pytest.raises(ValueError, match="dst dtype"):
            tile.scatter(
                ir.Var("dst", dst_type, span),
                ir.Var("src", src_type, span),
                ir.Var("idx", idx_type, span),
            )

    @pytest.mark.parametrize(
        ("dtype", "wrong_idx_dtype"),
        [
            (DataType.FP32, DataType.INT16),  # 4-byte dst requires INT32 idx
            (DataType.FP16, DataType.INT32),  # 2-byte dst requires INT16 idx
            (DataType.INT8, DataType.INT32),  # 1-byte dst requires INT16 idx
        ],
        ids=["fp32-needs-i32", "fp16-needs-i16", "i8-needs-i16"],
    )
    def test_tile_scatter_rejects_index_size_mismatch(self, dtype, wrong_idx_dtype):
        """tile.scatter enforces the dst-vs-indexes element-size rule."""
        span = ir.Span.unknown()
        src_type = ir.TileType(_const_dims(span, 4, 32), dtype)
        idx_type = ir.TileType(_const_dims(span, 4, 1), wrong_idx_dtype)
        dst_type = ir.TileType(_const_dims(span, 16, 32), dtype)

        with pytest.raises(ValueError, match="requires indexes dtype"):
            tile.scatter(
                ir.Var("dst", dst_type, span),
                ir.Var("src", src_type, span),
                ir.Var("idx", idx_type, span),
            )

    def test_tile_scatter_rejects_unsupported_dtype(self):
        """tile.scatter rejects element dtypes outside the spec whitelist."""
        span = ir.Span.unknown()
        src_type = ir.TileType(_const_dims(span, 4, 32), DataType.UINT32)
        idx_type = ir.TileType(_const_dims(span, 4, 1), DataType.INT32)
        dst_type = ir.TileType(_const_dims(span, 16, 32), DataType.UINT32)

        # dst is the first operand (DPS), so its dtype is validated first.
        with pytest.raises(ValueError, match="dst dtype"):
            tile.scatter(
                ir.Var("dst", dst_type, span),
                ir.Var("src", src_type, span),
                ir.Var("idx", idx_type, span),
            )

    def test_tile_scatter_allows_dst_col_mismatch(self):
        """tile.scatter's dst column count is independent of src (flat-addressed)."""
        span = ir.Span.unknown()
        src_type = ir.TileType(_const_dims(span, 4, 32), DataType.FP32)
        idx_type = ir.TileType(_const_dims(span, 4, 32), DataType.INT32)
        dst_type = ir.TileType(_const_dims(span, 16, 64), DataType.FP32)

        call = tile.scatter(
            ir.Var("dst", dst_type, span),
            ir.Var("src", src_type, span),
            ir.Var("idx", idx_type, span),
        )
        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        const_dims = [dim.value for dim in result_type.shape if isinstance(dim, ir.ConstInt)]
        assert const_dims == [16, 64]

    def test_tile_scatter_rejects_index_col_mismatch(self):
        """tile.scatter requires indexes.shape[1] == src.shape[1]."""
        span = ir.Span.unknown()
        src_type = ir.TileType(_const_dims(span, 4, 32), DataType.FP32)
        idx_type = ir.TileType(_const_dims(span, 4, 16), DataType.INT32)
        dst_type = ir.TileType(_const_dims(span, 16, 32), DataType.FP32)

        with pytest.raises(ValueError, match=r"indexes.shape\[1\] == src.shape\[1\]"):
            tile.scatter(
                ir.Var("dst", dst_type, span),
                ir.Var("src", src_type, span),
                ir.Var("idx", idx_type, span),
            )

    def test_tile_scatter_rejects_index_row_mismatch(self):
        """tile.scatter requires indexes.shape[0] == src.shape[0]."""
        span = ir.Span.unknown()
        src_type = ir.TileType(_const_dims(span, 4, 32), DataType.FP32)
        idx_type = ir.TileType(_const_dims(span, 8, 32), DataType.INT32)
        dst_type = ir.TileType(_const_dims(span, 16, 32), DataType.FP32)

        with pytest.raises(ValueError, match=r"indexes.shape\[0\] == src.shape\[0\]"):
            tile.scatter(
                ir.Var("dst", dst_type, span),
                ir.Var("src", src_type, span),
                ir.Var("idx", idx_type, span),
            )


class TestTileScatterMaskOps:
    """Test suite for tile.scatter_mask (mask form, DPS)."""

    @pytest.mark.parametrize(
        ("pattern", "src_cols", "dst_cols"),
        [
            (1, 8, 16),  # P0101 — stride 2
            (2, 8, 16),  # P1010 — stride 2
            (3, 4, 16),  # P0001 — stride 4
            (6, 4, 16),  # P1000 — stride 4
            (7, 16, 16),  # P1111 — no expansion
        ],
        ids=["P0101", "P1010", "P0001", "P1000", "P1111"],
    )
    def test_tile_scatter_mask_valid(self, pattern, src_cols, dst_cols):
        """tile.scatter_mask returns a tile aliased to dst with expanded cols."""
        span = ir.Span.unknown()
        src_type = ir.TileType(_const_dims(span, 4, src_cols), DataType.FP32)
        dst_type = ir.TileType(_const_dims(span, 4, dst_cols), DataType.FP32)

        call = tile.scatter_mask(
            ir.Var("dst", dst_type, span),
            ir.Var("src", src_type, span),
            pattern,
        )

        assert isinstance(call, ir.Call)
        assert call.op.name == "tile.scatter_mask"
        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        const_dims = [dim.value for dim in result_type.shape if isinstance(dim, ir.ConstInt)]
        assert const_dims == [4, dst_cols]

    def test_tile_scatter_mask_rejects_invalid_pattern(self):
        """tile.scatter_mask requires mask_pattern in [1, 7]."""
        span = ir.Span.unknown()
        src_type = ir.TileType(_const_dims(span, 4, 8), DataType.FP32)
        dst_type = ir.TileType(_const_dims(span, 4, 16), DataType.FP32)

        with pytest.raises(ValueError, match=r"mask_pattern in range \[1, 7\]"):
            tile.scatter_mask(
                ir.Var("dst", dst_type, span),
                ir.Var("src", src_type, span),
                42,
            )

    def test_tile_scatter_mask_rejects_col_expansion_mismatch(self):
        """tile.scatter_mask requires dst.cols == src.cols * stride."""
        span = ir.Span.unknown()
        src_type = ir.TileType(_const_dims(span, 4, 8), DataType.FP32)
        # P0101 stride is 2, dst should be 16 not 24
        dst_type = ir.TileType(_const_dims(span, 4, 24), DataType.FP32)

        with pytest.raises(ValueError, match="mask_pattern=1"):
            tile.scatter_mask(
                ir.Var("dst", dst_type, span),
                ir.Var("src", src_type, span),
                1,
            )

    def test_tile_scatter_mask_rejects_dtype_mismatch(self):
        """tile.scatter_mask requires dst and src to have the exact same dtype.

        Equal bit width is not sufficient — FP16 and INT16 are both 16-bit but
        the scatter spec mandates identical element types (no reinterpretation).
        """
        span = ir.Span.unknown()
        src_type = ir.TileType(_const_dims(span, 4, 8), DataType.FP16)
        dst_type = ir.TileType(_const_dims(span, 4, 16), DataType.INT16)

        with pytest.raises(ValueError, match="same dtype"):
            tile.scatter_mask(
                ir.Var("dst", dst_type, span),
                ir.Var("src", src_type, span),
                1,
            )


class TestTileConcatOps:
    """Test suite for tile.concat operation."""

    def test_tile_concat(self):
        """Test tile.concat operator - concatenate two tiles along columns."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                b: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[32, 16], pl.FP32] = pl.load(a, [0, 0], [32, 16])
                tile_b: pl.Tile[[32, 16], pl.FP32] = pl.load(b, [0, 0], [32, 16])
                tile_out: pl.Tile[[32, 32], pl.FP32] = pl.concat(tile_a, tile_b)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_out, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.concat" in ir_str

    def test_tile_concat_ir_level(self):
        """Test tile.concat at IR level with type deduction."""
        span = ir.Span.unknown()

        dim32 = ir.ConstInt(32, DataType.INT32, span)
        dim16 = ir.ConstInt(16, DataType.INT32, span)
        t0_type = ir.TileType([dim32, dim16], DataType.FP32)
        t1_type = ir.TileType([dim32, dim16], DataType.FP32)
        t0_var = ir.Var("src0", t0_type, span)
        t1_var = ir.Var("src1", t1_type, span)

        call = tile.concat(t0_var, t1_var)

        assert isinstance(call, ir.Call)
        assert call.op.name == "tile.concat"
        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        assert result_type.dtype == DataType.FP32
        assert len(result_type.shape) == 2
        # Output cols = 16 + 16 = 32
        assert isinstance(result_type.shape[1], ir.ConstInt)
        assert result_type.shape[1].value == 32

    # ------------------------------------------------------------------
    # valid_shape propagation: concat's union is a rectangle only
    # when both operands agree on valid rows AND src0 is fully valid in the concat
    # (column) dimension. Otherwise it is REJECTED, never widened.
    # ------------------------------------------------------------------

    def test_tile_concat_fully_valid_canonicalizes(self):
        """Two fully-valid operands -> fully-valid result (canonicalizes to no view)."""
        result_type = tile.concat(_mk_tile("a", [32, 16]), _mk_tile("b", [32, 16])).type
        assert isinstance(result_type, ir.TileType)
        assert _valid_ints(result_type) == [32, 32]
        assert result_type.tile_view is None

    def test_tile_concat_partial_final_operand(self):
        """The final operand may be partially valid in columns: A full [32, 16],
        B valid [32, 8] -> valid [32, 16 + 8 = 24] (B's trailing padding is excluded)."""
        result_type = tile.concat(_mk_tile("a", [32, 16]), _mk_tile("b", [32, 16], [32, 8])).type
        assert _valid_ints(result_type) == [32, 24]

    def test_tile_concat_dynamic_final_operand_symbolic(self):
        """A dynamic valid col extent on the final operand yields a symbolic Add."""
        vn = _index_scalar("vn")
        result_type = tile.concat(_mk_tile("a", [32, 16]), _mk_tile("b", [32, 16], [32, vn])).type
        assert isinstance(result_type, ir.TileType)
        assert result_type.tile_view is not None
        valid = result_type.tile_view.valid_shape
        assert isinstance(valid[0], ir.ConstInt) and valid[0].value == 32
        # cols = 16 + vn — a symbolic Add, not folded to a ConstInt.
        assert isinstance(valid[1], ir.Add)

    def test_tile_concat_rejects_partial_first_operand(self):
        """src0 partially valid in the concat dim leaves a gap -> rejected."""
        with pytest.raises(ValueError, match="partially valid along the concatenation"):
            tile.concat(_mk_tile("a", [32, 16], [32, 8]), _mk_tile("b", [32, 16]))

    def test_tile_concat_rejects_valid_row_mismatch(self):
        """Operands disagreeing on the valid row extent -> L-shaped union -> rejected."""
        with pytest.raises(ValueError, match="disagree on the valid row extent"):
            tile.concat(_mk_tile("a", [32, 16], [20, 16]), _mk_tile("b", [32, 16], [10, 16]))

    @pytest.mark.parametrize(
        ("t0_shape", "t0_dtype", "t1_shape", "t1_dtype", "match"),
        [
            ([32, 16], DataType.FP32, [32, 16], DataType.FP16, "same dtype"),
            ([32, 16], DataType.FP32, [8, 16], DataType.FP32, "row count must match"),
        ],
        ids=["dtype_mismatch", "row_mismatch"],
    )
    def test_tile_concat_rejects_invalid(self, t0_shape, t0_dtype, t1_shape, t1_dtype, match):
        """tile.concat enforces matching dtype and matching row counts."""
        span = ir.Span.unknown()
        t0_type = ir.TileType(_const_dims(span, *t0_shape), t0_dtype)
        t1_type = ir.TileType(_const_dims(span, *t1_shape), t1_dtype)

        with pytest.raises(ValueError, match=match):
            tile.concat(ir.Var("src0", t0_type, span), ir.Var("src1", t1_type, span))


class TestTileFormatShapeError:
    """Regression tests for issue #824: FormatShape prints readable shapes, not pointer addresses."""

    @staticmethod
    def _make_dim(span, value):
        """Create a dim that is either a ConstInt (from ``int``) or a symbolic Var (from ``str``)."""
        if isinstance(value, str):
            return ir.Var(value, ir.ScalarType(DataType.INT32), span)
        return ir.ConstInt(value, DataType.INDEX, span)

    @pytest.mark.parametrize(
        ("op_callable", "lhs_dims", "rhs_dims", "match"),
        [
            # Static shape mismatch surfaces the concrete dims (not pointers).
            (tile.add, [16, 16], [32, 16], r"\[16, 16\].*\[32, 16\]"),
            (tile.mul, [8, 16], [32, 16], r"\[8, 16\].*\[32, 16\]"),
            # Symbolic mismatch surfaces variable names instead of dim addresses.
            (tile.add, ["M", 16], ["N", 16], r"\[M, 16\].*\[N, 16\]"),
        ],
        ids=[
            "add_shape_mismatch_shows_readable_dims",
            "mul_shape_mismatch_shows_readable_dims",
            "add_symbolic_shape_mismatch_shows_var_names",
        ],
    )
    def test_tile_shape_mismatch_message(self, op_callable, lhs_dims, rhs_dims, match):
        """Shape-mismatch errors render dims/symbols as readable text (regression for #824)."""
        span = ir.Span.unknown()
        lhs_type = ir.TileType([self._make_dim(span, d) for d in lhs_dims], DataType.FP32)
        rhs_type = ir.TileType([self._make_dim(span, d) for d in rhs_dims], DataType.FP32)
        tile_a = ir.Var("a", lhs_type, span)
        tile_b = ir.Var("b", rhs_type, span)

        with pytest.raises(ValueError, match=match):
            op_callable(tile_a, tile_b)


class TestTileCiOp:
    """Tests for tile.ci (contiguous integer sequence generation, pto.tci)."""

    def test_tile_ci_ascending(self):
        """tile.ci returns a TileType with requested shape / dtype."""
        call = tile.ci(0, [1, 32], dtype=DataType.INT32)
        t = call.type
        assert isinstance(t, ir.TileType)
        assert t.dtype == DataType.INT32
        assert len(t.shape) == 2
        assert "tile.ci" in str(call)
        assert "descending=False" in str(call)

    def test_tile_ci_descending_kwarg_printed(self):
        """descending=True should appear in the printed IR."""
        call = tile.ci(10, [1, 16], dtype=DataType.INT32, descending=True)
        assert "descending=True" in str(call)

    def test_tile_ci_rejects_float_dtype(self):
        with pytest.raises(ValueError, match=r"INT16.*INT32.*UINT16.*UINT32"):
            tile.ci(0, [1, 32], dtype=DataType.FP32)

    def test_tile_ci_accepts_uint_dtype(self):
        call = tile.ci(0, [1, 16], dtype=DataType.UINT32)
        assert call is not None

    def test_tile_ci_rejects_cols_equal_one(self):
        with pytest.raises(ValueError, match="innermost dimension"):
            tile.ci(0, [32, 1], dtype=DataType.INT32)

    def test_tile_ci_rejects_multi_row_shape(self):
        """pto.tci only populates the first row, so leading dims must be 1."""
        with pytest.raises(ValueError, match=r"leading dimensions must be 1"):
            tile.ci(0, [4, 32], dtype=DataType.INT32)

    def test_tile_ci_rejects_start_dtype_mismatch(self):
        span = ir.Span.unknown()
        start = ir.Var("s", ir.ScalarType(DataType.INT16), span)
        with pytest.raises(ValueError, match=r"start.*dtype"):
            tile.ci(start, [1, 32], dtype=DataType.INT32)

    def test_tile_arange_alias_is_ci(self):
        assert pl.tile.arange is pl.tile.ci


class TestTileRandomOp:
    """tile.random (pto.trandom): counter-based RNG generator."""

    def test_tile_random_default(self):
        """tile.random returns a TileType with requested shape and UINT32 dtype."""
        call = tile.random(1, 2, 3, 4, 5, 6, [4, 256])
        t = call.type
        assert isinstance(t, ir.TileType)
        assert t.dtype == DataType.UINT32
        assert len(t.shape) == 2
        rows, cols = t.shape[0], t.shape[1]
        assert isinstance(rows, ir.ConstInt) and rows.value == 4
        assert isinstance(cols, ir.ConstInt) and cols.value == 256
        assert "tile.random" in str(call)

    def test_tile_random_int32_dtype(self):
        call = tile.random(1, 2, 3, 4, 5, 6, [8, 128], dtype=DataType.INT32)
        assert isinstance(call.type, ir.TileType)
        assert call.type.dtype == DataType.INT32

    def test_tile_random_rounds7(self):
        """rounds=7 must be preserved on the op, not silently dropped to the default 10."""
        call = tile.random(1, 2, 3, 4, 5, 6, [4, 64], rounds=7)
        assert "rounds=7" in str(call)

    def test_tile_random_valid_shape(self):
        """valid_shape narrows the written region; physical shape stays full."""
        call = tile.random(1, 2, 3, 4, 5, 6, [16, 128], valid_shape=[10, 80])
        t = call.type
        assert isinstance(t, ir.TileType)
        rows, cols = t.shape[0], t.shape[1]
        assert isinstance(rows, ir.ConstInt) and rows.value == 16
        assert isinstance(cols, ir.ConstInt) and cols.value == 128
        view = t.get_effective_tile_view()
        vr, vc = view.valid_shape[0], view.valid_shape[1]
        assert isinstance(vr, ir.ConstInt) and vr.value == 10
        assert isinstance(vc, ir.ConstInt) and vc.value == 80

    def test_tile_random_rejects_valid_shape_gt_shape(self):
        with pytest.raises(ValueError, match="valid_shape element"):
            tile.random(1, 2, 3, 4, 5, 6, [16, 128], valid_shape=[20, 80])

    def test_tile_random_rejects_float_dtype(self):
        with pytest.raises(ValueError, match=r"INT32.*UINT32"):
            tile.random(1, 2, 3, 4, 5, 6, [4, 64], dtype=DataType.FP32)

    def test_tile_random_rejects_bad_rounds(self):
        with pytest.raises(ValueError, match="rounds to be 7 or 10"):
            tile.random(1, 2, 3, 4, 5, 6, [4, 64], rounds=5)


class TestTileStoreDistributedDest:
    """``tile.store`` accepts ``DistributedTensorType`` as the destination.

    N6 stage-in pattern: a kernel writes a local tile into its own
    window-bound DistributedTensor slice (e.g. allreduce Phase 1 in
    tests/st/distributed/test_l3_allreduce.py). The verifier reaches
    DistributedTensorType through AsTensorTypeLike since
    DistributedTensorType inherits from TensorType but carries its own
    ObjectKind — exact-match As<TensorType>() would miss it.
    """

    def test_pl_store_into_distributed_tensor_parses(self):
        """``pl.store(tile, [0], dist_tensor)`` parses and types as the dst."""
        import pypto.language.distributed as pld  # noqa: PLC0415

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def stage_in(
                self,
                src: pl.Tensor[[64], pl.FP32],
                dst: pld.DistributedTensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                local: pl.Tile[[64], pl.FP32] = pl.load(src, [0], [64])
                return pl.store(local, [0], dst)

        ir_str = str(Program)
        assert "tile.store" in ir_str

    def test_tile_store_rejects_non_tensor_dst(self):
        """Regression: a Tile destination is still rejected by the verifier."""

        with pytest.raises(Exception, match="requires third argument to be a TensorType"):

            @pl.program
            class _Program:
                @pl.function(type=pl.FunctionType.InCore)
                def main(
                    self,
                    a: pl.Tensor[[128, 128], pl.FP32],
                ) -> pl.Tensor[[128, 128], pl.FP32]:
                    tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(a, [0, 0], [32, 32])
                    # Wrong: dst must be a (Distributed)TensorType, not a Tile.
                    return pl.store(tile_a, [0, 0], tile_a)  # pyright: ignore[reportArgumentType]


class TestTileLoadDistributedSrc:
    """``tile.load`` accepts ``DistributedTensorType`` as the source.

    Symmetric to ``tile.store``'s DistributedTensor dst: a kernel locally
    loads its own window-bound slice (e.g. read back a signal cell after a
    ``pld.system.wait`` barrier, as in
    tests/st/distributed/test_l3_notify_wait.py). The verifier reaches
    DistributedTensorType through AsTensorTypeLike since it inherits from
    TensorType but carries its own ObjectKind — exact-match As<TensorType>()
    would miss it.
    """

    def test_pl_load_from_distributed_tensor_parses(self):
        """``pl.load(dist_tensor, [0], [64])`` parses and types as a Tile."""
        import pypto.language.distributed as pld  # noqa: PLC0415

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def read_back(
                self,
                src: pld.DistributedTensor[[64], pl.FP32],
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                local: pl.Tile[[64], pl.FP32] = pl.load(src, [0], [64])
                return pl.store(local, [0], out)

        ir_str = str(Program)
        assert "tile.load" in ir_str

    def test_tile_load_rejects_non_tensor_src(self):
        """Regression: a non-tensor (Tile) source is still rejected."""

        with pytest.raises(Exception, match="requires first argument to be a TensorType"):

            @pl.program
            class _Program:
                @pl.function(type=pl.FunctionType.InCore)
                def main(
                    self,
                    a: pl.Tensor[[128, 128], pl.FP32],
                ) -> pl.Tensor[[32, 32], pl.FP32]:
                    tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(a, [0, 0], [32, 32])
                    # Wrong: load source must be a (Distributed)TensorType, not a Tile.
                    return pl.load(tile_a, [0, 0], [32, 32])  # pyright: ignore[reportArgumentType, reportReturnType]


class TestTileTransposeView:
    """tile.transpose_view: zero-copy fractal-layout reinterpretation (issue #1776)."""

    # (in_blayout, in_slayout, out_blayout, out_slayout, name). The transpose dual
    # flips each axis' major-ness independently (row<->col), leaving none_box fixed:
    # NZ<->ZN, NN<->ZZ, ND<->DN. A naive swap of the two fields would be wrong for
    # NN/ZZ (unchanged) and ND/DN (illegal none_box blayout).
    _DUALS = [
        ("NZ->ZN", "col_major", "row_major", "row_major", "col_major"),
        ("ZN->NZ", "row_major", "col_major", "col_major", "row_major"),
        ("NN->ZZ", "col_major", "col_major", "row_major", "row_major"),
        ("ZZ->NN", "row_major", "row_major", "col_major", "col_major"),
        ("ND->DN", "row_major", "none_box", "col_major", "none_box"),
        ("DN->ND", "col_major", "none_box", "row_major", "none_box"),
    ]

    @pytest.mark.parametrize(("name", "bin_", "sin", "bout", "sout"), _DUALS)
    def test_transpose_view_duality(self, name, bin_, sin, bout, sout):
        span = ir.Span.unknown()
        src_view = pl.TileView(blayout=getattr(pl.TileLayout, bin_), slayout=getattr(pl.TileLayout, sin))
        # [8, 16] -> transposed view is [16, 8].
        src_type = ir.TileType([8, 16], DataType.FP32, None, src_view)
        src = ir.Var("src", src_type, span)

        result_type = tile.transpose_view(src).type
        assert isinstance(result_type, ir.TileType)
        # Trailing two dims are swapped.
        assert isinstance(result_type.shape[0], ir.ConstInt) and result_type.shape[0].value == 16
        assert isinstance(result_type.shape[1], ir.ConstInt) and result_type.shape[1].value == 8
        # Each layout axis flips its major-ness (none_box stays none_box). The
        # default (row_major, none_box) = ND view canonicalizes to tile_view=None,
        # so read the effective layout.
        tv = result_type.tile_view
        eff_blayout = tv.blayout if tv is not None else pl.TileLayout.row_major
        eff_slayout = tv.slayout if tv is not None else pl.TileLayout.none_box
        assert eff_blayout == getattr(pl.TileLayout, bout)
        assert eff_slayout == getattr(pl.TileLayout, sout)

    def test_transpose_view_rejects_1d(self):
        span = ir.Span.unknown()
        src = ir.Var("src", ir.TileType([16], DataType.FP32), span)
        with pytest.raises(ValueError, match="at least 2 dimensions"):
            tile.transpose_view(src)


class TestTileSortValidShape:
    """valid_shape handling for sort ops.

    Sorting mixes padding into the valid region, so a partially-valid sort input is
    rejected rather than propagated. A fully-valid input keeps today's output shape.
    """

    def test_tile_sort32_fully_valid(self):
        """A fully-valid sort32 input keeps the doubled-last-dim output valid_shape."""
        result_type = tile.sort32(
            _mk_tile("s", [4, 32], dtype=DataType.FP32),
            _mk_tile("i", [4, 32], dtype=DataType.INT32),
        ).type
        assert isinstance(result_type, ir.TileType)
        assert _valid_ints(result_type) == [4, 64]

    def test_tile_sort32_rejects_partial_source(self):
        """A partially-valid src is rejected — padding would migrate into the sorted run."""
        with pytest.raises(ValueError, match="partially-valid sort input"):
            tile.sort32(
                _mk_tile("s", [4, 32], [4, 16], dtype=DataType.FP32),
                _mk_tile("i", [4, 32], dtype=DataType.INT32),
            )

    def test_tile_mrgsort_format1_rejects_partial_source(self):
        """mrgsort_format1 also rejects a partially-valid input."""
        with pytest.raises(ValueError, match="partially-valid sort input"):
            tile.mrgsort_format1(_mk_tile("s", [4, 128], [4, 64], dtype=DataType.FP32), 64)

    def test_tile_mrgsort_format2_rejects_partial_src0(self):
        """mrgsort_format2 rejects a partially-valid src0 (the src0 arm, sort.cpp).

        The 2-way form is (src0, src1, tmp); src0 is partial, src1 fully valid, so the
        non-src0 loop passes and the dedicated src0 check must fire.
        """
        with pytest.raises(ValueError, match="partially-valid sort input"):
            tile.mrgsort_format2(
                _mk_tile("s0", [4, 128], [4, 64], dtype=DataType.FP32),
                _mk_tile("s1", [4, 128], dtype=DataType.FP32),
                _mk_tile("tmp", [4, 256], dtype=DataType.FP32),
            )

    def test_tile_mrgsort_format2_rejects_partial_src1(self):
        """mrgsort_format2 rejects a partially-valid non-src0 operand (the src1..N-1
        loop arm, distinct from the src0 arm): src0 fully valid, src1 partial."""
        with pytest.raises(ValueError, match="partially-valid sort input"):
            tile.mrgsort_format2(
                _mk_tile("s0", [4, 128], dtype=DataType.FP32),
                _mk_tile("s1", [4, 128], [4, 64], dtype=DataType.FP32),
                _mk_tile("tmp", [4, 256], dtype=DataType.FP32),
            )


class TestValidDimTile:
    """pl.valid_dim — compile-time type query on tile operands."""

    def test_valid_dim_tile_explicit(self):
        """pl.valid_dim on a tile with an explicit valid_shape returns that extent."""
        tw = pl.Tile(expr=_mk_tile("t", [16, 32], [6, 20]))
        d0 = pl.valid_dim(tw, 0).unwrap()
        assert isinstance(d0, ir.ConstInt) and d0.value == 6
        d1 = pl.valid_dim(tw, 1).unwrap()
        assert isinstance(d1, ir.ConstInt) and d1.value == 20

    def test_valid_dim_tile_unset_returns_physical(self):
        """pl.valid_dim on a fully-valid tile returns the physical dim (design D2)."""
        tw = pl.Tile(expr=_mk_tile("t", [16, 32]))
        d0 = pl.valid_dim(tw, 0).unwrap()
        assert isinstance(d0, ir.ConstInt) and d0.value == 16
        d1 = pl.valid_dim(tw, 1).unwrap()
        assert isinstance(d1, ir.ConstInt) and d1.value == 32

    def test_valid_dim_tile_dynamic_returns_same_expr(self):
        """pl.valid_dim returns the exact valid_shape Expr (a runtime scalar Var) as-is."""
        span = ir.Span.unknown()
        vm = ir.Var("vm", ir.ScalarType(DataType.INDEX), span)
        tw = pl.Tile(expr=_mk_tile("t", [128, 128], [vm, 64]))
        assert pl.valid_dim(tw, 0).unwrap() is vm

    def test_valid_dim_tile_out_of_range_raises(self):
        """A negative or out-of-range axis raises (no numpy-style wrap-around)."""
        tw = pl.Tile(expr=_mk_tile("t", [16, 32]))
        with pytest.raises(IndexError, match="out of range"):
            pl.valid_dim(tw, 2)
        with pytest.raises(IndexError, match="out of range"):
            pl.valid_dim(tw, -1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
