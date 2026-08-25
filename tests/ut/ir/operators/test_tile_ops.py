# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for tile operations."""

import inspect
import math
from typing import Any, cast

import pypto.language as pl
import pytest
from pypto import DataType, ir
from pypto.ir.op import tile
from pypto.language.parser.diagnostics import InvalidOperationError

_OP_TILE_EXTRACT = ir.get_op("tile.extract").name
_OP_TILE_FILLPAD_EXPAND = ir.get_op("tile.fillpad_expand").name
_OP_TILE_MSCATTER = ir.get_op("tile.mscatter").name
_OP_TILE_ROW_EXPAND_ADD = ir.get_op("tile.row_expand_add").name
_OP_TILE_SET_VALIDSHAPE = ir.get_op("tile.set_validshape").name
_OP_TILE_SLICE = ir.get_op("tile.slice").name
_OP_TILE_TRANSPOSE = ir.get_op("tile.transpose").name


def _const_int(expr: ir.Expr) -> int:
    """The value of a constant Expr, asserting it is one."""
    assert isinstance(expr, ir.ConstInt), f"expected a ConstInt, got {type(expr).__name__}"
    return expr.value


def _operand_dtype(expr: ir.Expr) -> DataType:
    """Return a constant operand's dtype, narrowing ``Expr`` for the type checker."""
    assert isinstance(expr, (ir.ConstInt, ir.ConstFloat)), f"expected a constant, got {type(expr).__name__}"
    return expr.dtype


def _tile_result_dtype(call: ir.Call) -> DataType:
    """Return a tile call's result element dtype, narrowing ``Type``."""
    result_type = call.type
    assert isinstance(result_type, ir.TileType)
    return result_type.dtype


def _partial_tile(shape, valid_shape, pad=ir.PadValue.null, name="src", **view_kwargs):
    """A tile Var whose tile_view narrows it to `valid_shape`.

    Extra keyword arguments (blayout, slayout, fractal) are forwarded to the
    TileView, for callers that need a source with a specific layout.
    """
    span = ir.Span.unknown()
    view = ir.TileView(valid_shape=valid_shape, stride=[], start_offset=None, pad=pad, **view_kwargs)
    return ir.Var(name, ir.TileType(shape, DataType.FP32, tile_view=view), span)


def _valid_of(result_type):
    """Effective valid extents. GetEffectiveTileView always resolves them, so
    an absent view needs no fallback here."""
    return [
        d.value if isinstance(d, ir.ConstInt) else d
        for d in result_type.get_effective_tile_view().valid_shape
    ]


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

    def test_tile_div_precision_kwarg_and_scalar_dispatch(self):
        """Only tile-tile division carries the tdiv precision attribute."""
        span = ir.Span.unknown()
        tile_type = ir.TileType([8, 8], DataType.FP32)
        lhs = ir.Var("lhs", tile_type, span)
        rhs = ir.Var("rhs", tile_type, span)

        default_call = tile.div(lhs, rhs)
        high_precision_call = tile.div(lhs, rhs, high_precision=True)
        scalar_call = tile.div(lhs, 2.0)

        assert dict(default_call.kwargs) == {}
        assert dict(high_precision_call.kwargs) == {"high_precision": True}
        assert scalar_call.op.name == ir.get_op("tile.divs").name
        assert dict(scalar_call.kwargs) == {}
        with pytest.raises(TypeError, match=r"requires a Tile rhs"):
            tile.div(lhs, 2.0, high_precision=True)

    def test_tile_div_rejects_integer_high_precision_template_gap(self):
        """Do not expose the integer path that the PTOAS high-precision template cannot implement."""
        span = ir.Span.unknown()
        lhs = ir.Var("lhs", ir.TileType([8, 8], DataType.INT32), span)
        rhs = ir.Var("rhs", ir.TileType([8, 8], DataType.INT32), span)

        with pytest.raises(ValueError, match=r"high_precision only for FP16 or FP32"):
            tile.div(lhs, rhs, high_precision=True)

    @pytest.mark.parametrize("dtype", [DataType.INT16, DataType.INT32, DataType.FP16, DataType.FP32])
    def test_tile_div_accepts_ptoas_dtype_union(self, dtype):
        """tile.div accepts the union of the current A2/A3 and A5 contracts."""
        span = ir.Span.unknown()
        lhs = ir.Var("lhs", ir.TileType([8, 16], dtype), span)
        rhs = ir.Var("rhs", ir.TileType([8, 16], dtype), span)

        call = tile.div(lhs, rhs)

        assert _tile_result_dtype(call) == dtype

    def test_tile_div_rejects_mixed_dtypes(self):
        """PTOAS tdiv requires src0, src1, and dst to use one exact dtype."""
        span = ir.Span.unknown()
        lhs = ir.Var("lhs", ir.TileType([8, 16], DataType.FP16), span)
        rhs = ir.Var("rhs", ir.TileType([8, 16], DataType.FP32), span)

        with pytest.raises(ValueError, match=r"same dtype"):
            tile.div(lhs, rhs)

    def test_tile_div_rejects_different_physical_shapes_with_matching_valid_shape(self):
        """PTO-ISA TDIV templates require one common physical tile type."""
        span = ir.Span.unknown()
        view = ir.TileView(
            valid_shape=[7, 12],
            blayout=ir.TileLayout.row_major,
            slayout=ir.TileLayout.none_box,
        )
        lhs = ir.Var("lhs", ir.TileType([8, 16], DataType.FP32, tile_view=view), span)
        rhs = ir.Var("rhs", ir.TileType([10, 20], DataType.FP32, tile_view=view), span)

        with pytest.raises(ValueError, match=r"same physical shape"):
            tile.div(lhs, rhs)

    def test_tile_div_rejects_mismatched_valid_shapes(self):
        """Equal physical buffers are insufficient when valid extents differ."""
        span = ir.Span.unknown()
        lhs_view = ir.TileView(
            valid_shape=[7, 16],
            blayout=ir.TileLayout.row_major,
            slayout=ir.TileLayout.none_box,
        )
        rhs_view = ir.TileView(
            valid_shape=[8, 16],
            blayout=ir.TileLayout.row_major,
            slayout=ir.TileLayout.none_box,
        )
        lhs = ir.Var("lhs", ir.TileType([8, 16], DataType.FP32, tile_view=lhs_view), span)
        rhs = ir.Var("rhs", ir.TileType([8, 16], DataType.FP32, tile_view=rhs_view), span)

        with pytest.raises(ValueError, match=r"same valid_shape"):
            tile.div(lhs, rhs)

    def test_tile_div_rejects_unsupported_dtype(self):
        """INT8 is not in the current pto.tdiv dtype union."""
        span = ir.Span.unknown()
        lhs = ir.Var("lhs", ir.TileType([8, 16], DataType.INT8), span)
        rhs = ir.Var("rhs", ir.TileType([8, 16], DataType.INT8), span)

        with pytest.raises(ValueError, match=r"INT16, INT32, FP16, FP32"):
            tile.div(lhs, rhs)

    def test_tile_precision_apis_keep_positional_span_compatibility(self):
        """The pre-existing third/second positional argument remains ``span``."""
        span = ir.Span("tile_precision_compat.py", 7, 3, 7, 21)
        lhs = ir.Var("lhs", ir.TileType([8, 16], DataType.FP32), span)
        rhs = ir.Var("rhs", ir.TileType([8, 16], DataType.FP32), span)

        calls = (
            tile.div(lhs, rhs, span),
            tile.log(lhs, span),
            tile.recip(lhs, span),
        )

        assert all(call.span.filename == "tile_precision_compat.py" for call in calls)
        assert all(call.span.begin_line == 7 for call in calls)
        assert all(dict(call.kwargs) == {} for call in calls)

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

    @pytest.mark.parametrize("high_precision", [False, True])
    def test_tile_log_rejects_integer_contract(self, high_precision):
        """PTOAS does not define either logarithm precision mode for integer tiles."""
        span = ir.Span.unknown()
        src = ir.Var("src", ir.TileType([8, 8], DataType.INT32), span)

        with pytest.raises(ValueError, match=r"requires an FP16 or FP32"):
            tile.log(src, high_precision=high_precision)

    @pytest.mark.parametrize("dtype", [DataType.FP16, DataType.FP32])
    @pytest.mark.parametrize("high_precision", [False, True])
    def test_tile_log_accepts_supported_dtypes(self, dtype, high_precision):
        """Both precision modes preserve each PTOAS-supported float dtype."""
        span = ir.Span.unknown()
        src = ir.Var("src", ir.TileType([8, 8], dtype), span)

        call = tile.log(src, high_precision=high_precision)
        assert isinstance(call.type, ir.TileType)
        assert call.type.dtype == dtype
        expected_kwargs = {"high_precision": True} if high_precision else {}
        assert dict(call.kwargs) == expected_kwargs

    @pytest.mark.parametrize("dtype", [DataType.FP16, DataType.FP32])
    @pytest.mark.parametrize("high_precision", [False, True])
    def test_tile_recip_contract_and_precision(self, dtype, high_precision):
        """Both reciprocal precision modes preserve each supported float dtype."""
        span = ir.Span.unknown()
        src = ir.Var("src", ir.TileType([8, 8], dtype), span)

        call = tile.recip(src, high_precision=high_precision)

        assert isinstance(call.type, ir.TileType)
        assert call.type.dtype == dtype
        expected_kwargs = {"high_precision": True} if high_precision else {}
        assert dict(call.kwargs) == expected_kwargs

    @pytest.mark.parametrize("dtype", [DataType.INT32, DataType.BF16])
    def test_tile_recip_rejects_unsupported_high_precision_dtype(self, dtype):
        """The PTOAS high-precision reciprocal template only supports FP16 and FP32 inputs."""
        span = ir.Span.unknown()
        src = ir.Var("src", ir.TileType([8, 8], dtype), span)

        with pytest.raises(ValueError, match=r"high_precision only for FP16 or FP32"):
            tile.recip(src, high_precision=True)

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
        assert call.op.name == ir.get_op("tile.sin").name

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
        assert call.op.name == ir.get_op("tile.cos").name

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

    def test_tile_cast_requires_mode_kwarg(self):
        """tile.cast must reject a missing `mode` kwarg at construction time.

        `mode` is a declared attr that codegen reads unconditionally when emitting
        `pto.tcvt {rmode = ...}`. A missing kwarg reads back as 0 (round_mode NONE)
        instead of the DSL default ROUND, so DeduceTileCastType rejects it rather
        than letting a mode-less cast reach the backend.
        """
        span = ir.Span.unknown()
        src_type = ir.TileType(
            [ir.ConstInt(8, DataType.INT32, span), ir.ConstInt(16, DataType.INT32, span)],
            DataType.INT32,
        )
        src_var = ir.Var("src", src_type, span)

        with pytest.raises(ValueError, match="requires a 'mode' kwarg"):
            ir.create_op_call("tile.cast", [src_var], {"target_type": DataType.INT16}, span)

        # The canonical DSL constructor injects mode="round" — it must still work.
        call = tile.cast(src_var, DataType.INT16)
        assert dict(call.kwargs)["mode"] == 2

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

    def test_tile_unary_fully_valid_input_yields_no_explicit_view(self):
        """A fully valid input produces a fully valid result — canonicalized to no valid_shape."""
        span = ir.Span.unknown()
        src_type = ir.TileType(
            [ir.ConstInt(8, DataType.INT32, span), ir.ConstInt(16, DataType.INT32, span)],
            DataType.FP32,
        )
        src_var = ir.Var("src", src_type, span)

        result_type = tile.exp(src_var).type
        assert isinstance(result_type, ir.TileType)
        # Redundant full validity is canonicalized away, so there is nothing to print.
        assert result_type.tile_view is None or len(result_type.tile_view.valid_shape) == 0

    def test_tile_unary_result_carries_no_source_alias_metadata(self):
        """A fresh result aliases no source allocation: no memref, stride, or start offset."""
        sliced = self._make_sliced_tile_with_valid_shape()

        result_type = tile.neg(sliced).type
        assert isinstance(result_type, ir.TileType)
        assert result_type.memref is None
        assert result_type.tile_view is not None
        assert len(result_type.tile_view.stride) == 0
        assert result_type.tile_view.start_offset is None


class TestTileReductionOps:
    """Test suite for tile-level reduction operators."""

    def test_tile_row_max(self, ascend_backend, default_pass_manager):
        """Test tile.row_max operation."""

        @pl.program
        class RowMaxKernel:
            @pl.function(type=pl.FunctionType.InCore)
            def row_max_kernel(
                self, input: pl.Tensor[[128, 128], pl.FP32], output: pl.Tensor[[128, 1], pl.FP32]
            ) -> pl.Tensor[[128, 1], pl.FP32]:
                tile_in: pl.Tile[[32, 128], pl.FP32] = pl.load(input, [0, 0], [32, 128])
                tmp_tile: pl.Tile[[32, 128], pl.FP32] = pl.tile.create(
                    [32, 128], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
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
                tmp_tile: pl.Tile[[32, 128], pl.FP32] = pl.tile.create(
                    [32, 128], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
                )
                tile_sum: pl.Tile[[32, 1], pl.FP32] = pl.row_sum(tile_in, tmp_tile)
                result: pl.Tensor[[128, 1], pl.FP32] = pl.store(tile_sum, [0, 0], output)
                return result

        optimized_program = default_pass_manager.run_passes(RowSumKernel)

        assert optimized_program is not None
        assert "tile.row_sum" in str(optimized_program)

    @pytest.mark.parametrize("op", [tile.row_max, tile.row_sum, tile.row_min])
    @pytest.mark.parametrize("tmp_shape", [[8, 1], [8, 64], [8, 256]])
    def test_tile_row_reduction_rejects_undersized_tmp(self, op, tmp_shape):
        """Row reductions reject scratch storage that cannot hold the input tile."""
        span = ir.Span.unknown()
        input_tile = ir.Var("input_tile", ir.TileType([8, 512], DataType.FP32), span)
        tmp_tile = ir.Var("tmp_tile", ir.TileType(tmp_shape, DataType.FP32), span)

        with pytest.raises(ValueError, match="requires tmp_tile shape to be at least the input shape"):
            op(input_tile, tmp_tile)

    @pytest.mark.parametrize("op", [tile.row_max, tile.row_sum, tile.row_min])
    @pytest.mark.parametrize("tmp_shape", [[8, 512], [8, 640]])
    def test_tile_row_reduction_accepts_sufficient_tmp(self, op, tmp_shape):
        """Row reductions accept exact-size and padded scratch storage."""
        span = ir.Span.unknown()
        input_tile = ir.Var("input_tile", ir.TileType([8, 512], DataType.FP32), span)
        tmp_tile = ir.Var("tmp_tile", ir.TileType(tmp_shape, DataType.FP32), span)

        call = op(input_tile, tmp_tile)

        assert isinstance(call.type, ir.TileType)
        assert len(call.type.shape) == 2
        assert isinstance(call.type.shape[0], ir.ConstInt)
        assert isinstance(call.type.shape[1], ir.ConstInt)
        assert [call.type.shape[0].value, call.type.shape[1].value] == [8, 1]

    @pytest.mark.parametrize(
        "op", [tile.row_max, tile.row_sum, tile.row_min, tile.row_argmax, tile.row_argmin]
    )
    def test_tile_row_reduction_rejects_mismatched_tmp_dtype(self, op):
        """Row reductions require scratch storage with the input element type."""
        span = ir.Span.unknown()
        input_tile = ir.Var("input_tile", ir.TileType([8, 512], DataType.FP32), span)
        tmp_tile = ir.Var("tmp_tile", ir.TileType([8, 512], DataType.FP16), span)

        with pytest.raises(ValueError, match="requires tmp_tile dtype to match input dtype"):
            op(input_tile, tmp_tile)

    @pytest.mark.parametrize("op", [tile.row_argmax, tile.row_argmin])
    @pytest.mark.parametrize("tmp_shape", [[8, 256], [8, 640]])
    def test_tile_row_arg_reduction_rejects_non_exact_tmp_shape(self, op, tmp_shape):
        """Row arg reductions reject both undersized and oversized scratch storage."""
        span = ir.Span.unknown()
        input_tile = ir.Var("input_tile", ir.TileType([8, 512], DataType.FP32), span)
        tmp_tile = ir.Var("tmp_tile", ir.TileType(tmp_shape, DataType.FP32), span)

        with pytest.raises(ValueError, match="requires tmp_tile shape to exactly match the input shape"):
            op(input_tile, tmp_tile)

    @pytest.mark.parametrize("op", [tile.row_argmax, tile.row_argmin])
    def test_tile_row_arg_reduction_accepts_exact_tmp_shape(self, op):
        """Row arg reductions accept scratch storage matching the input shape."""
        span = ir.Span.unknown()
        input_tile = ir.Var("input_tile", ir.TileType([8, 512], DataType.FP32), span)
        tmp_tile = ir.Var("tmp_tile", ir.TileType([8, 512], DataType.FP32), span)

        call = op(input_tile, tmp_tile)

        assert isinstance(call.type, ir.TileType)
        assert call.type.dtype == DataType.INT32

    @pytest.mark.parametrize("dtype", [DataType.INT16, DataType.INT32, DataType.FP16, DataType.FP32])
    def test_tile_row_min_accepts_exact_pto_contract(self, dtype):
        """tile.row_min accepts every PTO dtype and produces a DN column vector."""
        span = ir.Span.unknown()
        input_tile = ir.Var("input_tile", ir.TileType([8, 32], dtype), span)
        tmp_tile = ir.Var("tmp_tile", ir.TileType([8, 32], dtype), span)

        call = tile.row_min(input_tile, tmp_tile)

        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        assert result_type.dtype == dtype
        assert [dim.value for dim in result_type.shape if isinstance(dim, ir.ConstInt)] == [8, 1]
        result_view = result_type.get_effective_tile_view()
        assert result_view.blayout == ir.TileLayout.col_major
        assert result_view.slayout == ir.TileLayout.none_box

    @pytest.mark.parametrize("dtype", [DataType.INT8, DataType.BF16])
    def test_tile_row_min_rejects_unsupported_dtype(self, dtype):
        """tile.row_min rejects dtypes outside the PTO TROWMIN contract."""
        span = ir.Span.unknown()
        input_tile = ir.Var("input_tile", ir.TileType([8, 32], dtype), span)
        tmp_tile = ir.Var("tmp_tile", ir.TileType([8, 32], dtype), span)

        with pytest.raises(ValueError, match=r"requires input dtype in \{INT16, INT32, FP16, FP32\}"):
            tile.row_min(input_tile, tmp_tile)

    @pytest.mark.parametrize(
        ("blayout", "slayout"),
        [
            pytest.param(ir.TileLayout.col_major, ir.TileLayout.none_box, id="col-major-flat"),
            pytest.param(ir.TileLayout.row_major, ir.TileLayout.col_major, id="row-major-boxed"),
        ],
    )
    def test_tile_row_min_rejects_non_nd_input_layout(self, blayout, slayout):
        """tile.row_min requires an effective row-major, non-boxed source view."""
        span = ir.Span.unknown()
        input_view = ir.TileView(blayout=blayout, slayout=slayout)
        input_tile = ir.Var(
            "input_tile",
            ir.TileType([8, 32], DataType.FP32, tile_view=input_view),
            span,
        )
        tmp_tile = ir.Var("tmp_tile", ir.TileType([8, 32], DataType.FP32), span)

        with pytest.raises(ValueError, match=r"requires an ND input layout"):
            tile.row_min(input_tile, tmp_tile)

    def test_tile_row_min_does_not_constrain_tmp_layout(self):
        """TROWMIN tmp keeps the existing same-dtype/rank/size safety subset only."""
        span = ir.Span.unknown()
        input_tile = ir.Var("input_tile", ir.TileType([8, 32], DataType.FP32), span)
        boxed_tmp_view = ir.TileView(blayout=ir.TileLayout.col_major, slayout=ir.TileLayout.row_major)
        tmp_tile = ir.Var(
            "tmp_tile",
            ir.TileType([8, 64], DataType.FP32, tile_view=boxed_tmp_view),
            span,
        )

        call = tile.row_min(input_tile, tmp_tile)

        assert isinstance(call.type, ir.TileType)
        assert call.type.get_effective_tile_view().blayout == ir.TileLayout.col_major

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

    def _make_row_tmp_var(self):
        """Helper: the scratch tile the row reductions take as their second argument.

        The PTO row-reduction instructions use it as full-size scratch, so it matches the
        input's physical shape rather than the reduced output shape.
        """
        span = ir.Span.unknown()
        tmp_type = ir.TileType(
            [ir.ConstInt(8, DataType.INT32, span), ir.ConstInt(32, DataType.INT32, span)],
            DataType.FP32,
        )
        return ir.Var("tmp", tmp_type, span)

    def test_tile_row_sum_inherits_input_valid_shape(self):
        """tile.row_sum output valid_shape must mirror input on the kept dim (issue #1401)."""
        sliced = self._make_sliced_tile_with_valid_shape(valid_rows=4, valid_cols=32)
        span = ir.Span.unknown()
        tmp_type = ir.TileType(
            [ir.ConstInt(8, DataType.INT32, span), ir.ConstInt(32, DataType.INT32, span)],
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
            [ir.ConstInt(8, DataType.INT32, span), ir.ConstInt(32, DataType.INT32, span)],
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

    # ------------------------------------------------------------------
    # Reducing a partially valid axis is accepted: the backend reduction
    # kernels bound their loop by the source's valid extent (TRowSum reads
    # srcTile.GetValidCol(), TColSum reads GetValidRow()), so they fold exactly
    # the real cells and never read padding. The reduced axis therefore
    # collapses to a fully valid output axis. An *empty* valid extent is the one
    # input those kernels reject, so it is caught here instead.
    # ------------------------------------------------------------------

    def test_tile_row_sum_partial_reduced_axis_collapses_to_valid(self):
        """Reducing a partially valid axis folds only the real cells (16 of 32 cols)."""
        # valid_cols=16 of 32: the *reduced* axis is partial.
        sliced = self._make_sliced_tile_with_valid_shape(valid_rows=8, valid_cols=16)

        result_type = tile.row_sum(sliced, self._make_row_tmp_var()).type
        assert isinstance(result_type, ir.TileType)
        assert _const_values(result_type.shape) == [8, 1]
        # The kernel folded exactly the 16 real cells into one output cell, so the partial
        # extent does not leak into the result: every cell of the [8, 1] output is real.
        # Full validity is canonical, so no explicit valid_shape survives.
        assert result_type.tile_view is None or len(result_type.tile_view.valid_shape) == 0

    def test_tile_row_sum_symbolic_reduced_axis_accepted(self):
        """A symbolic (unproved) valid extent on the reduced axis is accepted, not rejected."""
        span = ir.Span.unknown()
        vlen = ir.Var("vlen", ir.ScalarType(DataType.INDEX), span)
        src_type = ir.TileType(
            [ir.ConstInt(8, DataType.INT32, span), ir.ConstInt(32, DataType.INT32, span)],
            DataType.FP32,
            None,
            ir.TileView(valid_shape=[ir.ConstInt(8, DataType.INDEX, span), vlen]),
        )
        src_var = ir.Var("src", src_type, span)

        result_type = tile.row_sum(src_var, self._make_row_tmp_var()).type
        assert isinstance(result_type, ir.TileType)
        assert _const_values(result_type.shape) == [8, 1]
        # `vlen` is never proved equal to 32, yet the reduction is still well defined: the
        # kernel reduces whatever the runtime valid extent turns out to be. The result is
        # fully valid either way, so no symbolic extent survives into the output type.
        assert result_type.tile_view is None or len(result_type.tile_view.valid_shape) == 0

    def test_tile_row_sum_rejects_empty_valid_extent(self):
        """A provably zero valid extent has no real data to reduce and is rejected."""
        empty = self._make_sliced_tile_with_valid_shape(valid_rows=8, valid_cols=0)

        with pytest.raises(ValueError, match="valid extent on axis 1 is 0"):
            tile.row_sum(empty, self._make_row_tmp_var())

    def test_tile_col_sum_rejects_empty_valid_extent(self):
        """The guard covers the reduced *row* axis of a column reduction too."""
        empty = self._make_sliced_tile_with_valid_shape(valid_rows=0, valid_cols=32)

        with pytest.raises(ValueError, match="valid extent on axis 0 is 0"):
            tile.col_sum(empty)

    def test_tile_reduction_rejects_unsigned_empty_valid_extent(self):
        """An empty extent is rejected whatever its dtype's signedness.

        The extent proof only decides operands of matching signedness, so an unsigned zero
        compared against a signed zero is merely "unknown". A constant zero must therefore be
        recognised by value, or exactly the empty region this guard exists for slips through.
        """
        span = ir.Span.unknown()
        src_var = ir.Var(
            "src",
            ir.TileType(
                [ir.ConstInt(8, DataType.INT32, span), ir.ConstInt(32, DataType.INT32, span)],
                DataType.FP32,
                None,
                ir.TileView(
                    valid_shape=[
                        ir.ConstInt(8, DataType.INDEX, span),
                        ir.ConstInt(0, DataType.UINT64, span),  # unsigned zero
                    ]
                ),
            ),
            span,
        )
        tmp_var = self._make_row_tmp_var()

        with pytest.raises(ValueError, match="valid extent on axis 1 is 0"):
            tile.row_sum(src_var, tmp_var)

    def test_tile_reduction_rejects_rank_mismatched_valid_shape(self):
        """A valid_shape whose rank differs from the physical shape is rejected, not read past."""
        span = ir.Span.unknown()
        bad = ir.Var(
            "src",
            ir.TileType(
                [ir.ConstInt(8, DataType.INT32, span), ir.ConstInt(32, DataType.INT32, span)],
                DataType.FP32,
                None,
                ir.TileView(valid_shape=[4]),  # rank 1 against a rank-2 tile
            ),
            span,
        )

        with pytest.raises(ValueError, match="valid_shape rank"):
            tile.col_sum(bad)


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

    def test_tile_row_expand_add_accepts_row_major_packed_vector(self):
        """The packed vector may have larger physical extents than its valid region."""
        span = ir.Span.unknown()
        main = ir.Var("main", ir.TileType([8, 32], DataType.FP32), span)
        packed_view = ir.TileView(
            valid_shape=[8, 8],
            blayout=ir.TileLayout.row_major,
            slayout=ir.TileLayout.none_box,
        )
        packed_row = ir.Var(
            "packed_row",
            ir.TileType([10, 16], DataType.FP32, tile_view=packed_view),
            span,
        )

        call = tile.row_expand_add(main, packed_row)
        assert call.op.name == _OP_TILE_ROW_EXPAND_ADD
        assert len(call.args) == 2

    def test_tile_row_expand_add_rejects_invalid_packed_valid_width_and_tmp_type(self):
        """Packed valid rows must be 32 bytes and tmp must be a TileType."""
        span = ir.Span.unknown()
        main = ir.Var("main", ir.TileType([8, 32], DataType.FP32), span)
        bad_view = ir.TileView(
            valid_shape=[8, 7],
            blayout=ir.TileLayout.row_major,
            slayout=ir.TileLayout.none_box,
        )
        bad_row = ir.Var(
            "bad_row",
            ir.TileType([8, 16], DataType.FP32, tile_view=bad_view),
            span,
        )

        with pytest.raises(ValueError, match=r"last dimension to be 8"):
            tile.row_expand_add(main, bad_row)

        row = ir.Var("row", ir.TileType([8, 1], DataType.FP32), span)
        bad_tmp = ir.ConstFloat(0.0, DataType.FP32, span)
        with pytest.raises(ValueError, match=r"tmp to be a TileType"):
            tile.row_expand_add(main, row, tmp=bad_tmp)

    def test_tile_row_expand_add_rejects_mixed_dtypes(self):
        """PTOAS requires src0, src1, and dst to have one exact dtype."""
        span = ir.Span.unknown()
        main = ir.Var("main", ir.TileType([8, 32], DataType.FP32), span)
        row = ir.Var("row", ir.TileType([8, 1], DataType.FP16), span)

        with pytest.raises(ValueError, match=r"src0 and src1 to have the same dtype"):
            tile.row_expand_add(main, row)

    def test_tile_row_expand_add_rejects_valid_row_mismatch(self):
        """Physical rows matching is insufficient when valid row extents differ."""
        span = ir.Span.unknown()
        main_view = ir.TileView(
            valid_shape=[6, 32],
            blayout=ir.TileLayout.row_major,
            slayout=ir.TileLayout.none_box,
        )
        row_view = ir.TileView(
            valid_shape=[5, 1],
            blayout=ir.TileLayout.col_major,
            slayout=ir.TileLayout.none_box,
        )
        main = ir.Var(
            "main",
            ir.TileType([8, 32], DataType.FP32, tile_view=main_view),
            span,
        )
        row = ir.Var(
            "row",
            ir.TileType([8, 1], DataType.FP32, tile_view=row_view),
            span,
        )

        with pytest.raises(ValueError, match=r"src1 valid row extent to match src0/dst"):
            tile.row_expand_add(main, row)

    def test_tile_row_expand_add_requires_provable_dynamic_carrier_extents(self):
        """Shared dynamic rows are valid; unrelated rows or widths are unsafe."""
        span = ir.Span.unknown()
        valid_rows = ir.Var("valid_rows", ir.ScalarType(DataType.INDEX), span)
        other_rows = ir.Var("other_rows", ir.ScalarType(DataType.INDEX), span)
        carrier_cols = ir.Var("carrier_cols", ir.ScalarType(DataType.INDEX), span)
        main_view = ir.TileView(
            valid_shape=[valid_rows, 32],
            blayout=ir.TileLayout.row_major,
            slayout=ir.TileLayout.none_box,
        )
        main = ir.Var("main", ir.TileType([8, 32], DataType.FP32, tile_view=main_view), span)

        matching_row_view = ir.TileView(
            valid_shape=[valid_rows, 1],
            blayout=ir.TileLayout.col_major,
            slayout=ir.TileLayout.none_box,
        )
        matching_row = ir.Var(
            "matching_row",
            ir.TileType([8, 1], DataType.FP32, tile_view=matching_row_view),
            span,
        )
        assert tile.row_expand_add(main, matching_row).op.name == _OP_TILE_ROW_EXPAND_ADD

        unrelated_row_view = ir.TileView(
            valid_shape=[other_rows, 1],
            blayout=ir.TileLayout.col_major,
            slayout=ir.TileLayout.none_box,
        )
        unrelated_row = ir.Var(
            "unrelated_row",
            ir.TileType([8, 1], DataType.FP32, tile_view=unrelated_row_view),
            span,
        )
        with pytest.raises(ValueError, match=r"src1 valid row extent to match src0/dst"):
            tile.row_expand_add(main, unrelated_row)

        dynamic_width_view = ir.TileView(
            valid_shape=[valid_rows, carrier_cols],
            blayout=ir.TileLayout.row_major,
            slayout=ir.TileLayout.none_box,
        )
        dynamic_width_row = ir.Var(
            "dynamic_width_row",
            ir.TileType([8, 8], DataType.FP32, tile_view=dynamic_width_view),
            span,
        )
        with pytest.raises(ValueError, match=r"valid last dimension to be 8"):
            tile.row_expand_add(main, dynamic_width_row)

    @pytest.mark.parametrize(
        "dtype",
        [DataType.INT8, DataType.INT16, DataType.INT32, DataType.FP16, DataType.FP32],
    )
    def test_tile_row_expand_add_accepts_ptoas_dtype_union(self, dtype):
        """The tile contract exposes the union of supported PTOAS architectures."""
        span = ir.Span.unknown()
        main = ir.Var("main", ir.TileType([8, 32], dtype), span)
        row = ir.Var("row", ir.TileType([8, 1], dtype), span)

        call = tile.row_expand_add(main, row)

        assert _tile_result_dtype(call) == dtype

    def test_tile_row_expand_add_rejects_non_row_major_src0(self):
        """PTOAS requires src0 and dst to use row-major block layout."""
        span = ir.Span.unknown()
        main_view = ir.TileView(
            valid_shape=[8, 32],
            blayout=ir.TileLayout.col_major,
            slayout=ir.TileLayout.none_box,
        )
        main = ir.Var("main", ir.TileType([8, 32], DataType.FP32, tile_view=main_view), span)
        row = ir.Var("row", ir.TileType([8, 1], DataType.FP32), span)

        with pytest.raises(ValueError, match=r"src0 effective blayout to be row_major"):
            tile.row_expand_add(main, row)

    def test_tile_row_expand_add_rejects_non_row_major_valid_col_mismatch(self):
        """A DN [M, 1] carrier must also have valid_shape[1] equal to one."""
        span = ir.Span.unknown()
        row_view = ir.TileView(
            valid_shape=[8, 0],
            blayout=ir.TileLayout.col_major,
            slayout=ir.TileLayout.none_box,
        )
        main = ir.Var("main", ir.TileType([8, 32], DataType.FP32), span)
        row = ir.Var("row", ir.TileType([8, 1], DataType.FP32, tile_view=row_view), span)

        with pytest.raises(ValueError, match=r"valid last dimension to be 1"):
            tile.row_expand_add(main, row)

    def test_tile_row_expand_add_rejects_unsupported_dtype(self):
        """BF16 is outside the current pto.trowexpandadd dtype union."""
        span = ir.Span.unknown()
        main = ir.Var("main", ir.TileType([8, 32], DataType.BF16), span)
        row = ir.Var("row", ir.TileType([8, 1], DataType.BF16), span)

        with pytest.raises(ValueError, match=r"INT8, INT16, INT32, FP16, FP32"):
            tile.row_expand_add(main, row)

    def test_tile_row_expand_add_keeps_positional_span_and_keyword_tmp(self):
        """The third positional slot stays span; tmp is keyword-only."""
        span = ir.Span("row_expand_add_compat.py", 11, 2, 11, 18)
        main = ir.Var("main", ir.TileType([8, 32], DataType.FP32), span)
        row = ir.Var("row", ir.TileType([8, 1], DataType.FP32), span)
        tmp = ir.Var("tmp", ir.TileType([8, 32], DataType.FP32), span)

        without_tmp = tile.row_expand_add(main, row, span)
        with_tmp = tile.row_expand_add(main, row, span, tmp=tmp)

        assert without_tmp.span.filename == "row_expand_add_compat.py"
        assert with_tmp.span.filename == "row_expand_add_compat.py"
        assert len(without_tmp.args) == 2
        assert len(with_tmp.args) == 3
        assert with_tmp.args[2] is tmp

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

    def test_matmul_and_acc_propagate_padded_operand_valid_shape(self):
        """Matmul uses logical valid extents inside box-aligned storage.

        A boundary Right tile may require 32 physical INT8 columns even when
        only 16 columns are in bounds.  The resulting Acc tile must retain the
        physical width for allocation and the logical width for computation
        and stores; ``matmul_acc`` must preserve both.
        """
        span = ir.Span.unknown()

        def dims(*values):
            return [ir.ConstInt(value, DataType.INDEX, span) for value in values]

        lhs_type = ir.TileType(
            dims(16, 128),
            DataType.INT8,
            tile_view=ir.TileView(valid_shape=dims(16, 128)),
            memory_space=ir.MemorySpace.Left,
        )
        rhs_type = ir.TileType(
            dims(128, 32),
            DataType.INT8,
            tile_view=ir.TileView(valid_shape=dims(128, 16)),
            memory_space=ir.MemorySpace.Right,
        )
        lhs = ir.Var("lhs", lhs_type, span)
        rhs = ir.Var("rhs", rhs_type, span)

        matmul_type = tile.matmul(lhs, rhs).type
        assert isinstance(matmul_type, ir.TileType)
        assert _const_values(matmul_type.shape) == [16, 32]
        assert _valid_of(matmul_type) == [16, 16]

        acc_type = ir.TileType(
            dims(16, 32),
            DataType.INT32,
            tile_view=ir.TileView(valid_shape=dims(16, 16)),
            memory_space=ir.MemorySpace.Acc,
        )
        acc = ir.Var("acc", acc_type, span)
        matmul_acc_type = tile.matmul_acc(acc, lhs, rhs).type
        assert isinstance(matmul_acc_type, ir.TileType)
        assert _const_values(matmul_acc_type.shape) == [16, 32]
        assert _valid_of(matmul_acc_type) == [16, 16]

    @staticmethod
    def _narrowable_matmul_operands(span, lhs_valid_rows):
        """lhs [64, 128] narrowed to `lhs_valid_rows`, plus a fully valid rhs / bias."""

        def dims(*values):
            return [ir.ConstInt(value, DataType.INDEX, span) for value in values]

        lhs = ir.Var(
            "lhs",
            ir.TileType(
                dims(64, 128),
                DataType.INT8,
                tile_view=ir.TileView(valid_shape=[lhs_valid_rows, ir.ConstInt(128, DataType.INDEX, span)]),
                memory_space=ir.MemorySpace.Left,
            ),
            span,
        )
        rhs = ir.Var(
            "rhs",
            ir.TileType(
                dims(128, 256),
                DataType.INT8,
                tile_view=ir.TileView(valid_shape=dims(128, 256)),
                memory_space=ir.MemorySpace.Right,
            ),
            span,
        )
        bias = ir.Var(
            "bias",
            ir.TileType(dims(1, 256), DataType.INT32, tile_view=ir.TileView(valid_shape=dims(1, 256))),
            span,
        )
        return lhs, rhs, bias

    @staticmethod
    def _compact_of(result_type):
        """Compact mode of a deduced tile type, narrowing ``Type`` for the checker."""
        assert isinstance(result_type, ir.TileType)
        return result_type.get_effective_tile_view().compact

    def test_matmul_family_stamps_compact_for_runtime_narrowed_rows(self):
        """Issue #2470: a runtime-narrowed L0C must advertise the stride ``mad`` wrote at.

        ``mad`` lays the product out with an N-fractal stride of ceil(M/16)*16,
        where M is the lhs *valid* rows, while every Acc reader keys off the
        tile's compile-time physical ``Rows`` unless the tile is compact.  A
        runtime row count therefore has to stamp compact — the same reasoning
        that makes a partial ``tile.extract`` into L0A/L0B compact (#2232).
        """
        span = ir.Span.unknown()
        rows = ir.Var("rows", ir.ScalarType(DataType.INDEX), span)
        lhs, rhs, bias = self._narrowable_matmul_operands(span, rows)

        matmul_type = tile.matmul(lhs, rhs).type
        assert isinstance(matmul_type, ir.TileType)
        assert _valid_of(matmul_type) == [rows, 256]
        assert self._compact_of(matmul_type) == ir.CompactMode.normal

        # The accumulate step reuses the accumulator's storage, so it must reach
        # the same compact mode or the two views of one L0C buffer disagree.
        acc = ir.Var("acc", matmul_type, span)
        assert self._compact_of(tile.matmul_acc(acc, lhs, rhs).type) == ir.CompactMode.normal
        assert self._compact_of(tile.matmul_bias(lhs, rhs, bias).type) == ir.CompactMode.normal

    def test_matmul_full_rows_stay_noncompact(self):
        """A fully valid accumulator keeps its historical non-compact form."""
        span = ir.Span.unknown()
        lhs, rhs, bias = self._narrowable_matmul_operands(span, ir.ConstInt(64, DataType.INDEX, span))

        matmul_type = tile.matmul(lhs, rhs).type
        assert self._compact_of(matmul_type) == ir.CompactMode.null

        acc = ir.Var("acc", matmul_type, span)
        assert self._compact_of(tile.matmul_acc(acc, lhs, rhs).type) == ir.CompactMode.null
        assert self._compact_of(tile.matmul_bias(lhs, rhs, bias).type) == ir.CompactMode.null

    def test_matmul_narrowed_columns_alone_stay_noncompact(self):
        """Only the row extent moves the Acc stride, so a narrow N changes nothing.

        Every Acc stride the ISA derives is a function of ``validRow`` alone; a
        narrowed column extent leaves writer and reader in agreement.
        """
        span = ir.Span.unknown()

        def dims(*values):
            return [ir.ConstInt(value, DataType.INDEX, span) for value in values]

        lhs = ir.Var(
            "lhs",
            ir.TileType(
                dims(64, 128),
                DataType.INT8,
                tile_view=ir.TileView(valid_shape=dims(64, 128)),
                memory_space=ir.MemorySpace.Left,
            ),
            span,
        )
        rhs = ir.Var(
            "rhs",
            ir.TileType(
                dims(128, 256),
                DataType.INT8,
                tile_view=ir.TileView(valid_shape=dims(128, 240)),
                memory_space=ir.MemorySpace.Right,
            ),
            span,
        )

        matmul_type = tile.matmul(lhs, rhs).type
        assert _valid_of(matmul_type) == [64, 240]
        assert self._compact_of(matmul_type) == ir.CompactMode.null

    def test_set_validshape_keeps_the_stride_an_accumulator_was_written_at(self):
        """Narrowing an already-written accumulator must not re-interpret its bytes.

        ``tile.set_validshape`` is a metadata-only alias that may run *after*
        the buffer was filled. A full-width ``tile.matmul`` lays L0C out at the
        physical row pitch, so deriving compact from the *new* valid rows here
        would make every later reader walk those same bytes at
        ``ceil(16/16)*16`` instead — silently corrupting them. The op inherits
        the source's compact mode and nothing else.
        """
        span = ir.Span.unknown()
        lhs, rhs, _ = self._narrowable_matmul_operands(span, ir.ConstInt(64, DataType.INDEX, span))
        written = tile.matmul(lhs, rhs).type
        assert self._compact_of(written) == ir.CompactMode.null

        narrowed = tile.set_validshape(ir.Var("acc", written, span), 16, 256).type
        assert _valid_of(narrowed) == [16, 256]
        assert self._compact_of(narrowed) == ir.CompactMode.null

    def test_set_validshape_carries_a_compact_accumulator_through(self):
        """A compact accumulator stays compact when its valid window moves."""
        span = ir.Span.unknown()
        rows = ir.Var("rows", ir.ScalarType(DataType.INDEX), span)
        lhs, rhs, _ = self._narrowable_matmul_operands(span, rows)
        written = tile.matmul(lhs, rhs).type
        assert self._compact_of(written) == ir.CompactMode.normal

        narrowed = tile.set_validshape(ir.Var("acc", written, span), rows, 256).type
        assert self._compact_of(narrowed) == ir.CompactMode.normal

    def test_matmul_acc_inherits_the_accumulator_compact_mode(self):
        """The in-place result must describe its buffer exactly as the input does.

        ``tile.matmul_acc`` is ``set_output_reuses_input(0)``; codegen only
        aliases result and accumulator when their ``TileBufSignature`` — compact
        included — matches, so the result inherits rather than re-derives.
        """
        span = ir.Span.unknown()
        rows = ir.Var("rows", ir.ScalarType(DataType.INDEX), span)
        lhs, rhs, _ = self._narrowable_matmul_operands(span, rows)

        # A non-compact accumulator stays non-compact even beside a narrowed lhs.
        plain_acc = ir.TileType(
            [ir.ConstInt(64, DataType.INDEX, span), ir.ConstInt(256, DataType.INDEX, span)],
            DataType.INT32,
            tile_view=ir.TileView(
                valid_shape=[rows, ir.ConstInt(256, DataType.INDEX, span)],
            ),
            memory_space=ir.MemorySpace.Acc,
        )
        result = tile.matmul_acc(ir.Var("acc", plain_acc, span), lhs, rhs).type
        assert self._compact_of(result) == ir.CompactMode.null

    def test_set_validshape_leaves_non_accumulator_tiles_noncompact(self):
        """A narrowed Vec tile has no L0C stride contract to preserve."""
        span = ir.Span.unknown()
        rows = ir.Var("rows", ir.ScalarType(DataType.INDEX), span)
        vec = ir.Var(
            "vec",
            ir.TileType(
                [ir.ConstInt(64, DataType.INDEX, span), ir.ConstInt(256, DataType.INDEX, span)],
                DataType.FP32,
                memory_space=ir.MemorySpace.Vec,
            ),
            span,
        )

        narrowed = tile.set_validshape(vec, rows, 256).type
        assert _valid_of(narrowed) == [rows, 256]
        assert self._compact_of(narrowed) == ir.CompactMode.null

    def test_matmul_rejects_mismatched_physical_k_with_matching_valid_k(self):
        """Logical K agreement does not make incompatible physical boxes legal."""
        span = ir.Span.unknown()

        def dims(*values):
            return [ir.ConstInt(value, DataType.INDEX, span) for value in values]

        lhs = ir.Var(
            "lhs",
            ir.TileType(dims(16, 32), DataType.INT8, tile_view=ir.TileView(valid_shape=dims(16, 16))),
            span,
        )
        rhs = ir.Var(
            "rhs",
            ir.TileType(dims(16, 32), DataType.INT8, tile_view=ir.TileView(valid_shape=dims(16, 16))),
            span,
        )

        with pytest.raises(ValueError, match="matching physical inner dimensions"):
            tile.matmul(lhs, rhs)

    def test_matmul_allows_rhs_valid_k_to_contain_lhs_valid_k(self):
        """PTO reads lhs valid K and permits a wider valid window on rhs."""
        span = ir.Span.unknown()

        def dims(*values):
            return [ir.ConstInt(value, DataType.INDEX, span) for value in values]

        lhs = ir.Var(
            "lhs",
            ir.TileType(
                dims(16, 256),
                DataType.FP16,
                tile_view=ir.TileView(valid_shape=dims(16, 255)),
            ),
            span,
        )
        rhs = ir.Var(
            "rhs",
            ir.TileType(
                dims(256, 16),
                DataType.FP16,
                tile_view=ir.TileView(valid_shape=dims(256, 16)),
            ),
            span,
        )

        result = tile.matmul(lhs, rhs).type
        assert isinstance(result, ir.TileType)
        assert _const_values(result.shape) == [16, 16]
        assert _valid_of(result) == [16, 16]

    def test_matmul_rejects_rhs_valid_k_smaller_than_lhs(self):
        """The rhs valid K window must contain every lhs K element PTO reads."""
        span = ir.Span.unknown()

        def dims(*values):
            return [ir.ConstInt(value, DataType.INDEX, span) for value in values]

        lhs = ir.Var(
            "lhs",
            ir.TileType(
                dims(16, 256),
                DataType.FP16,
                tile_view=ir.TileView(valid_shape=dims(16, 256)),
            ),
            span,
        )
        rhs = ir.Var(
            "rhs",
            ir.TileType(
                dims(256, 16),
                DataType.FP16,
                tile_view=ir.TileView(valid_shape=dims(255, 16)),
            ),
            span,
        )

        with pytest.raises(ValueError, match="rhs valid K to cover lhs valid K"):
            tile.matmul(lhs, rhs)

    def test_matmul_acc_allows_acc_valid_shape_to_contain_product(self):
        """PTO may update a smaller product rectangle inside a wider accumulator."""
        span = ir.Span.unknown()

        def dims(*values):
            return [ir.ConstInt(value, DataType.INDEX, span) for value in values]

        acc = ir.Var(
            "acc",
            ir.TileType(
                dims(16, 32),
                DataType.FP32,
                tile_view=ir.TileView(valid_shape=dims(16, 32)),
            ),
            span,
        )
        lhs = ir.Var(
            "lhs",
            ir.TileType(
                dims(16, 16),
                DataType.FP16,
                tile_view=ir.TileView(valid_shape=dims(16, 16)),
            ),
            span,
        )
        rhs = ir.Var(
            "rhs",
            ir.TileType(
                dims(16, 32),
                DataType.FP16,
                tile_view=ir.TileView(valid_shape=dims(16, 24)),
            ),
            span,
        )

        result = tile.matmul_acc(acc, lhs, rhs).type
        assert isinstance(result, ir.TileType)
        assert _const_values(result.shape) == [16, 32]
        assert _valid_of(result) == [16, 32]

    @pytest.mark.parametrize(
        ("acc_valid", "lhs_valid", "rhs_valid", "message"),
        [
            ((15, 32), (16, 16), (16, 24), "acc valid M"),
            ((16, 23), (16, 16), (16, 24), "acc valid N"),
            ((16, 32), (16, 16), (15, 24), "rhs valid K"),
        ],
    )
    def test_matmul_acc_rejects_valid_shape_not_containing_product(
        self, acc_valid, lhs_valid, rhs_valid, message
    ):
        """Each PTO-computed extent must fit its corresponding valid window."""
        span = ir.Span.unknown()

        def dims(*values):
            return [ir.ConstInt(value, DataType.INDEX, span) for value in values]

        acc = ir.Var(
            "acc",
            ir.TileType(dims(16, 32), DataType.FP32, tile_view=ir.TileView(valid_shape=dims(*acc_valid))),
            span,
        )
        lhs = ir.Var(
            "lhs",
            ir.TileType(dims(16, 16), DataType.FP16, tile_view=ir.TileView(valid_shape=dims(*lhs_valid))),
            span,
        )
        rhs = ir.Var(
            "rhs",
            ir.TileType(dims(16, 32), DataType.FP16, tile_view=ir.TileView(valid_shape=dims(*rhs_valid))),
            span,
        )

        with pytest.raises(ValueError, match=message):
            tile.matmul_acc(acc, lhs, rhs)

    @pytest.mark.parametrize(
        ("acc_shape", "lhs_shape", "rhs_shape", "message"),
        [
            ((32, 32), (16, 16), (16, 32), "physical M"),
            ((16, 64), (16, 16), (16, 32), "physical N"),
            ((16, 32), (16, 32), (16, 32), "physical K"),
        ],
    )
    def test_matmul_acc_rejects_mismatched_physical_boxes_with_matching_valid_shape(
        self, acc_shape, lhs_shape, rhs_shape, message
    ):
        """All three physical dimensions remain part of the matmul_acc contract."""
        span = ir.Span.unknown()

        def dims(*values):
            return [ir.ConstInt(value, DataType.INDEX, span) for value in values]

        valid_shape = dims(16, 16)
        acc = ir.Var(
            "acc",
            ir.TileType(dims(*acc_shape), DataType.INT32, tile_view=ir.TileView(valid_shape=valid_shape)),
            span,
        )
        lhs = ir.Var(
            "lhs",
            ir.TileType(dims(*lhs_shape), DataType.INT8, tile_view=ir.TileView(valid_shape=valid_shape)),
            span,
        )
        rhs = ir.Var(
            "rhs",
            ir.TileType(dims(*rhs_shape), DataType.INT8, tile_view=ir.TileView(valid_shape=valid_shape)),
            span,
        )

        with pytest.raises(ValueError, match=message):
            tile.matmul_acc(acc, lhs, rhs)

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
                tile_c: pl.Tile[[16, 32], pl.FP32] = pl.gemv(tile_a, tile_b)
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
                acc_in: pl.Tensor[[16, 128], pl.FP32],
                a: pl.Tensor[[1, 64], pl.FP32],
                b: pl.Tensor[[64, 128], pl.FP32],
                output: pl.Tensor[[1, 128], pl.FP32],
            ) -> pl.Tensor[[1, 128], pl.FP32]:
                tile_acc: pl.Tile[[16, 32], pl.FP32] = pl.load(acc_in, [0, 0], [16, 32], valid_shape=[1, 32])
                tile_a: pl.Tile[[1, 16], pl.FP32] = pl.load(a, [0, 0], [1, 16])
                tile_b: pl.Tile[[16, 32], pl.FP32] = pl.load(b, [0, 0], [16, 32])
                tile_c: pl.Tile[[16, 32], pl.FP32] = pl.gemv_acc(tile_acc, tile_a, tile_b)
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
                tile_c: pl.Tile[[16, 32], pl.FP32] = pl.gemv_bias(tile_a, tile_b, tile_bias)
                result: pl.Tensor[[1, 128], pl.FP32] = pl.store(tile_c, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.gemv_bias" in ir_str

    def test_tile_gemv_physical_accumulator_and_logical_valid_shape(self):
        """GEMV pads Acc rows to 16 while preserving the logical [1, N] extent."""
        span = ir.Span.unknown()
        lhs = ir.Var(
            "lhs",
            ir.TileType([1, 128], DataType.FP32, tile_view=ir.TileView(valid_shape=[1, 64])),
            span,
        )
        rhs = ir.Var(
            "rhs",
            ir.TileType([128, 128], DataType.FP32, tile_view=ir.TileView(valid_shape=[64, 48])),
            span,
        )
        bias = ir.Var(
            "bias",
            ir.TileType([1, 128], DataType.FP32, tile_view=ir.TileView(valid_shape=[1, 48])),
            span,
        )

        result = tile.gemv(lhs, rhs)
        result_type = result.type
        assert isinstance(result_type, ir.TileType)
        assert [d.value for d in result_type.shape if isinstance(d, ir.ConstInt)] == [16, 128]
        assert _valid_of(result_type) == [1, 48]

        acc_result = tile.gemv_acc(result, lhs, rhs)
        bias_result = tile.gemv_bias(lhs, rhs, bias)
        for call in (acc_result, bias_result):
            call_type = call.type
            assert isinstance(call_type, ir.TileType)
            assert [d.value for d in call_type.shape if isinstance(d, ir.ConstInt)] == [16, 128]
            assert _valid_of(call_type) == [1, 48]

    @pytest.mark.parametrize(
        ("input_dtype", "output_dtype"),
        [
            (DataType.INT8, DataType.INT32),
            (DataType.FP16, DataType.FP32),
            (DataType.BF16, DataType.FP32),
            (DataType.FP32, DataType.FP32),
        ],
    )
    def test_tile_gemv_family_accepts_supported_dtype_triples(self, input_dtype, output_dtype):
        span = ir.Span.unknown()
        lhs = ir.Var(
            "lhs",
            ir.TileType([1, 128], input_dtype, tile_view=ir.TileView(valid_shape=[1, 64])),
            span,
        )
        rhs = ir.Var(
            "rhs",
            ir.TileType([128, 128], input_dtype, tile_view=ir.TileView(valid_shape=[64, 48])),
            span,
        )
        acc = ir.Var(
            "acc",
            ir.TileType([16, 128], output_dtype, tile_view=ir.TileView(valid_shape=[1, 48])),
            span,
        )
        bias = ir.Var(
            "bias",
            ir.TileType([1, 128], output_dtype, tile_view=ir.TileView(valid_shape=[1, 48])),
            span,
        )

        for call in (tile.gemv(lhs, rhs), tile.gemv_acc(acc, lhs, rhs), tile.gemv_bias(lhs, rhs, bias)):
            result_type = call.type
            assert isinstance(result_type, ir.TileType)
            assert result_type.dtype == output_dtype

    def test_tile_gemv_rejects_insufficient_rhs_logical_k(self):
        span = ir.Span.unknown()
        lhs = ir.Var(
            "lhs",
            ir.TileType([1, 128], DataType.FP32, tile_view=ir.TileView(valid_shape=[1, 96])),
            span,
        )
        rhs = ir.Var(
            "rhs",
            ir.TileType([128, 128], DataType.FP32, tile_view=ir.TileView(valid_shape=[64, 48])),
            span,
        )

        with pytest.raises(ValueError, match="rhs valid K to cover lhs valid K"):
            tile.gemv(lhs, rhs)

    def test_tile_gemv_rejects_unsupported_input_dtype(self):
        span = ir.Span.unknown()
        lhs = ir.Var("lhs", ir.TileType([1, 128], DataType.INT16), span)
        rhs = ir.Var("rhs", ir.TileType([128, 128], DataType.INT16), span)

        with pytest.raises(ValueError, match="supports only INT8"):
            tile.gemv(lhs, rhs)

    def test_tile_gemv_rejects_mixed_input_dtypes(self):
        span = ir.Span.unknown()
        lhs = ir.Var("lhs", ir.TileType([1, 128], DataType.FP16), span)
        rhs = ir.Var("rhs", ir.TileType([128, 128], DataType.FP32), span)

        with pytest.raises(ValueError, match="identical lhs and rhs data types"):
            tile.gemv(lhs, rhs)

    def test_tile_gemv_bias_rejects_input_dtype_bias(self):
        span = ir.Span.unknown()
        lhs = ir.Var("lhs", ir.TileType([1, 128], DataType.FP16), span)
        rhs = ir.Var("rhs", ir.TileType([128, 128], DataType.FP16), span)
        bias = ir.Var("bias", ir.TileType([1, 128], DataType.FP16), span)

        with pytest.raises(ValueError, match="requires bias dtype fp32"):
            tile.gemv_bias(lhs, rhs, bias)

    def test_tile_gemv_acc_rejects_input_dtype_accumulator(self):
        span = ir.Span.unknown()
        lhs = ir.Var("lhs", ir.TileType([1, 128], DataType.FP16), span)
        rhs = ir.Var("rhs", ir.TileType([128, 128], DataType.FP16), span)
        acc = ir.Var("acc", ir.TileType([16, 128], DataType.FP16), span)

        with pytest.raises(ValueError, match="requires accumulator dtype fp32"):
            tile.gemv_acc(acc, lhs, rhs)

    def test_tile_gemv_rejects_multiple_logical_lhs_rows(self):
        span = ir.Span.unknown()
        lhs = ir.Var(
            "lhs",
            ir.TileType([1, 128], DataType.FP32, tile_view=ir.TileView(valid_shape=[2, 64])),
            span,
        )
        rhs = ir.Var(
            "rhs",
            ir.TileType([128, 128], DataType.FP32, tile_view=ir.TileView(valid_shape=[64, 48])),
            span,
        )

        with pytest.raises(ValueError, match="logical row extent to be exactly 1"):
            tile.gemv(lhs, rhs)

    def test_tile_gemv_rejects_padded_physical_lhs_rows(self):
        span = ir.Span.unknown()
        lhs = ir.Var(
            "lhs",
            ir.TileType([16, 128], DataType.FP32, tile_view=ir.TileView(valid_shape=[1, 64])),
            span,
        )
        rhs = ir.Var(
            "rhs",
            ir.TileType([128, 128], DataType.FP32, tile_view=ir.TileView(valid_shape=[64, 48])),
            span,
        )

        with pytest.raises(ValueError, match="physical row extent to be exactly 1"):
            tile.gemv(lhs, rhs)

    def test_tile_gemv_bias_rejects_undersized_valid_shape(self):
        span = ir.Span.unknown()
        lhs = ir.Var(
            "lhs",
            ir.TileType([1, 128], DataType.FP32, tile_view=ir.TileView(valid_shape=[1, 64])),
            span,
        )
        rhs = ir.Var(
            "rhs",
            ir.TileType([128, 128], DataType.FP32, tile_view=ir.TileView(valid_shape=[64, 48])),
            span,
        )
        bias = ir.Var(
            "bias",
            ir.TileType([1, 128], DataType.FP32, tile_view=ir.TileView(valid_shape=[1, 32])),
            span,
        )

        with pytest.raises(ValueError, match=r"bias valid N to cover output valid N=48"):
            tile.gemv_bias(lhs, rhs, bias)

    def test_tile_gemv_acc_rejects_mismatched_valid_shape(self):
        span = ir.Span.unknown()
        lhs = ir.Var(
            "lhs",
            ir.TileType([1, 128], DataType.FP32, tile_view=ir.TileView(valid_shape=[1, 64])),
            span,
        )
        rhs = ir.Var(
            "rhs",
            ir.TileType([128, 128], DataType.FP32, tile_view=ir.TileView(valid_shape=[64, 48])),
            span,
        )
        acc = ir.Var(
            "acc",
            ir.TileType([16, 128], DataType.FP32, tile_view=ir.TileView(valid_shape=[1, 32])),
            span,
        )

        with pytest.raises(ValueError, match="accumulator valid_shape"):
            tile.gemv_acc(acc, lhs, rhs)


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
        assert call.op.name == _OP_TILE_SLICE
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
        assert call.op.name == _OP_TILE_SLICE
        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        assert result_type.tile_view is not None
        assert len(result_type.shape) == 2
        assert isinstance(result_type.shape[1], ir.ConstInt)
        assert result_type.tile_view.valid_shape[1] is valid_n

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
        assert call.op.name == _OP_TILE_SLICE
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

    def test_tile_reshape_identity_keeps_the_whole_source_view(self):
        """An identity reshape is a view onto the same bytes, so it keeps the source view.

        Re-deriving the view from the shape gives the space-agnostic flat default, which
        ``NormalizeImplicitTileView`` rescues only for a view that collapses — and a
        narrowed, padded or compact Acc box never does, so the flat layout would stick
        (issue #2470). ``stride`` and ``start_offset`` are the same story one level down:
        they *are* the address arithmetic, and dropping them relocates a strided sub-view.
        """
        span = ir.Span.unknown()
        rows = ir.ConstInt(64, DataType.INT32, span)
        cols = ir.ConstInt(128, DataType.INT32, span)
        source_view = ir.TileView(
            valid_shape=[16, 128],
            stride=[256, 1],
            start_offset=512,
            blayout=ir.TileLayout.col_major,
            slayout=ir.TileLayout.row_major,
            fractal=1024,
            compact=ir.CompactMode.normal,
        )
        source_type = ir.TileType([rows, cols], DataType.INT32, None, source_view, ir.MemorySpace.Acc)
        source = ir.Var("acc", source_type, span)

        result = tile.reshape(source, [64, 128]).type
        assert isinstance(result, ir.TileType)
        view = result.tile_view
        assert view is not None

        assert result.memory_space == ir.MemorySpace.Acc
        assert view.blayout == ir.TileLayout.col_major
        assert view.slayout == ir.TileLayout.row_major
        assert view.fractal == 1024
        assert view.compact == ir.CompactMode.normal
        assert [_const_int(dim) for dim in view.stride] == [256, 1], (
            "an identity reshape must keep the source's stride — it is the same addressing"
        )
        assert view.start_offset is not None and _const_int(view.start_offset) == 512
        assert [_const_int(dim) for dim in view.valid_shape] == [16, 128]

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
        assert call.op.name == ir.get_op("tile.reshape").name
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
    # valid_shape mapping through reshape. A reshape is a zero-copy view, so it
    # cannot invent data: the result's valid region is the source's, expressed in
    # the target shape, and a region the target shape cannot spell as an
    # origin-anchored box is rejected rather than rounded up to fully valid.
    # tile.reshape and tensor.reshape share one rule, so ConvertTensorToTileOps
    # cannot rewrite a tensor.reshape into a tile.reshape that widens it back —
    # see the mirrored cases in test_tensor_ops.py.
    # ------------------------------------------------------------------

    def test_tile_reshape_fully_valid_input_yields_no_explicit_valid_shape(self):
        """A fully valid source stays fully valid, canonicalized to nothing to print."""
        span = ir.Span.unknown()
        dims = [ir.ConstInt(8, DataType.INT32, span), ir.ConstInt(16, DataType.INT32, span)]
        tile_var = ir.Var("src", ir.TileType(dims, DataType.FP32), span)

        result_type = tile.reshape(tile_var, [16, 8]).type

        assert isinstance(result_type, ir.TileType)
        assert result_type.tile_view is None or len(result_type.tile_view.valid_shape) == 0

    def test_tile_reshape_maps_row_prefix_to_target_rectangle(self):
        """Valid rows are a contiguous flat prefix: 5*16 = 80 cells = 10 rows of 8."""
        result_type = tile.reshape(_partial_tile([8, 16], [5, 16]), [16, 8]).type

        assert _valid_of(result_type) == [10, 8]

    def test_tile_reshape_preserves_compact_storage_mode(self):
        """A zero-copy reshape keeps the source tile's compact representation."""
        span = ir.Span.unknown()
        source = ir.Var(
            "src",
            ir.TileType(
                [8, 16],
                DataType.INT8,
                tile_view=ir.TileView(valid_shape=[5, 16], compact=ir.CompactMode.normal),
            ),
            span,
        )

        result_type = tile.reshape(source, [16, 8]).type

        assert isinstance(result_type, ir.TileType)
        assert _valid_of(result_type) == [10, 8]
        assert result_type.get_effective_tile_view().compact == ir.CompactMode.normal

    def test_tile_reshape_drops_full_unit_axis_exactly(self):
        """Erasing a provably full unit axis preserves an arbitrary rectangle."""
        result_type = tile.reshape(_partial_tile([1, 8, 16], [1, 8, 5]), [8, 16]).type

        assert _valid_of(result_type) == [8, 5]

    def test_tile_reshape_empty_region_stays_empty(self):
        """The empty set has an exact representation in every target shape."""
        result_type = tile.reshape(_partial_tile([8, 16], [0, 16]), [16, 8]).type

        assert _valid_of(result_type) == [0, 0]

    def test_tile_reshape_rejects_region_that_is_not_a_flat_prefix(self):
        """Valid columns leave gaps between real rows, so no target rectangle spans them."""
        with pytest.raises(ValueError, match="real data is scattered across the buffer"):
            tile.reshape(_partial_tile([8, 16], [8, 5]), [16, 8])

    def test_tile_reshape_rejects_prefix_without_a_target_rectangle(self):
        """80 cells is not a whole number of 32-wide rows, so [4, 32] cannot spell it."""
        with pytest.raises(ValueError, match="do not fill a whole number of rows"):
            tile.reshape(_partial_tile([8, 16], [5, 16]), [4, 32])

    @staticmethod
    def _col_major_tile(shape, valid_shape):
        span = ir.Span.unknown()
        view = ir.TileView(valid_shape=valid_shape, blayout=ir.TileLayout.col_major)
        return ir.Var("src", ir.TileType(shape, DataType.FP32, tile_view=view), span)

    def test_tile_reshape_rejects_partial_col_major_source(self):
        """Flat-prefix mapping reads row-major offsets, so a col_major source is rejected.

        A col_major [2, 3] valid [1, 3] really occupies flat cells {0, 2, 4};
        reading it row-major would return a box covering {0, 1, 2} and mark two
        padding elements as real — the widening this rule exists to prevent.
        """
        with pytest.raises(ValueError, match="not stored row-major"):
            tile.reshape(self._col_major_tile([2, 3], [1, 3]), [1, 6])

    def test_tile_reshape_maps_a_valid_region_of_a_different_int_dtype(self):
        """A full axis must read as full even when its dtype differs from the shape's.

        ``tile.set_validshape`` emits UINT64 extents while the physical shape is
        INDEX, and the analyzer only compares extents of matching signedness — so
        16 == 16 came back unknown, the axis read as partial, and this mappable
        reshape was rejected outright.
        """
        span = ir.Span.unknown()
        src = ir.Var("src", ir.TileType([8, 16], DataType.FP32), span)
        narrowed = tile.set_validshape(
            src, ir.ConstInt(5, DataType.UINT64, span), ir.ConstInt(16, DataType.UINT64, span)
        )

        result_type = tile.reshape(narrowed, [16, 8]).type

        assert _valid_of(result_type) == [10, 8]

    def test_tile_reshape_allows_fully_valid_col_major_source(self):
        """Only the flat-prefix case reads storage order; a full region never does."""
        result_type = tile.reshape(self._col_major_tile([2, 3], [2, 3]), [1, 6]).type

        assert _valid_of(result_type) == [1, 6]

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
        assert call.op.name == _OP_TILE_FILLPAD_EXPAND
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
        assert call.op.name == _OP_TILE_FILLPAD_EXPAND
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
        assert call.op.name == _OP_TILE_TRANSPOSE
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
        assert call.op.name == _OP_TILE_TRANSPOSE
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
        assert call.op.name == _OP_TILE_SET_VALIDSHAPE
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
        assert call.op.name == _OP_TILE_SET_VALIDSHAPE
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

        with pytest.raises(ValueError, match="must be >= 0"):
            tile.set_validshape(tile_var, -1, 8)

    def test_tile_set_validshape_rejects_exceeding_bound(self):
        """Valid dimensions exceeding physical shape are rejected."""
        span = ir.Span.unknown()

        dim16 = ir.ConstInt(16, DataType.INT32, span)
        tile_type = ir.TileType([dim16, dim16], DataType.FP32)
        tile_var = ir.Var("tile", tile_type, span)

        with pytest.raises(ValueError, match="exceeds tile bound"):
            tile.set_validshape(tile_var, 32, 8)

    def test_transform_operators_registered(self):
        """Test that transform operators are registered."""
        assert ir.is_op_registered("tile.slice")
        assert ir.is_op_registered("tile.reshape")
        assert ir.is_op_registered("tile.reinterpret_view")
        assert ir.is_op_registered("tile.transpose")
        assert ir.is_op_registered("tile.set_validshape")


class TestTileReinterpretViewIR:
    """IR semantics for tile.reinterpret_view before public DSL lowering."""

    @staticmethod
    def _var(
        shape: list[int],
        dtype: DataType,
        view: ir.TileView | None = None,
    ) -> ir.Var:
        return ir.Var("src", ir.TileType(shape, dtype, tile_view=view), ir.Span.unknown())

    @staticmethod
    def _shape_values(result_type: ir.TileType) -> list[int]:
        return [dim.value for dim in result_type.shape if isinstance(dim, ir.ConstInt)]

    def test_auto_shape_uses_row_major_contiguous_axis(self):
        call = tile.reinterpret_view(self._var([8, 16], DataType.FP32), DataType.INT16)

        assert call.op.name == ir.get_op("tile.reinterpret_view").name
        assert isinstance(call.type, ir.TileType)
        assert call.type.dtype == DataType.INT16
        assert self._shape_values(call.type) == [8, 32]

    def test_rank_one_auto_shape_uses_its_only_axis(self):
        call = tile.reinterpret_view(self._var([16], DataType.FP32), DataType.INT16)

        assert isinstance(call.type, ir.TileType)
        assert self._shape_values(call.type) == [32]

    def test_auto_shape_uses_col_major_contiguous_axis(self):
        view = ir.TileView(blayout=ir.TileLayout.col_major, slayout=ir.TileLayout.none_box)
        call = tile.reinterpret_view(self._var([8, 16], DataType.FP32, view), DataType.INT16)

        assert isinstance(call.type, ir.TileType)
        assert self._shape_values(call.type) == [16, 16]
        assert call.type.get_effective_tile_view().blayout == ir.TileLayout.col_major

    def test_explicit_byte_equivalent_shape(self):
        call = tile.reinterpret_view(
            self._var([8, 16], DataType.FP32),
            DataType.INT16,
            shape=[4, 64],
        )

        assert isinstance(call.type, ir.TileType)
        assert self._shape_values(call.type) == [4, 64]

    def test_wider_dtype_requires_divisible_contiguous_extent(self):
        with pytest.raises(ValueError, match=r"dimension 1 .*not divisible by 2"):
            tile.reinterpret_view(self._var([8, 15], DataType.INT16), DataType.FP32)

    def test_partial_valid_shape_scales_with_auto_shape(self):
        view = ir.TileView(valid_shape=[4, 12])
        call = tile.reinterpret_view(self._var([8, 16], DataType.FP32, view), DataType.INT16)

        assert isinstance(call.type, ir.TileType)
        result_view = call.type.get_effective_tile_view()
        assert [dim.value for dim in result_view.valid_shape if isinstance(dim, ir.ConstInt)] == [4, 24]

    @pytest.mark.parametrize(
        ("source_pad", "expected_pad"),
        [
            (ir.PadValue.null, ir.PadValue.null),
            (ir.PadValue.zero, ir.PadValue.zero),
            (ir.PadValue.max, ir.PadValue.null),
            (ir.PadValue.min, ir.PadValue.null),
        ],
    )
    def test_normalizes_dtype_dependent_padding(self, source_pad, expected_pad):
        view = ir.TileView(pad=source_pad)

        call = tile.reinterpret_view(self._var([8, 16], DataType.FP32, view), DataType.INT16)

        assert isinstance(call.type, ir.TileType)
        assert call.type.get_effective_tile_view().pad == expected_pad

    def test_rejects_mismatched_explicit_byte_size(self):
        with pytest.raises(ValueError, match=r"equal source and target byte sizes.*512 bytes.*256 bytes"):
            tile.reinterpret_view(
                self._var([8, 16], DataType.FP32),
                DataType.INT16,
                shape=[8, 16],
            )

    def test_rejects_same_dtype(self):
        with pytest.raises(ValueError, match="requires source and target dtypes to differ"):
            tile.reinterpret_view(self._var([8, 16], DataType.FP32), DataType.FP32)

    def test_rejects_boxed_tile(self):
        boxed = ir.TileView(blayout=ir.TileLayout.col_major, slayout=ir.TileLayout.row_major)
        with pytest.raises(ValueError, match="only supports flat tiles"):
            tile.reinterpret_view(self._var([8, 16], DataType.FP32, boxed), DataType.INT16)


class TestTileReinterpretViewDSL:
    """Public ``pl.tile.reinterpret_view`` wrapper and export coverage."""

    @staticmethod
    def _tile() -> pl.Tile:
        source = ir.Var("src", ir.TileType([8, 16], DataType.FP32), ir.Span.unknown())
        return pl.Tile(expr=source)

    def test_auto_shape_wrapper(self):
        result = pl.tile.reinterpret_view(self._tile(), pl.INT16)

        assert isinstance(result, pl.Tile)
        call = result.unwrap()
        assert isinstance(call, ir.Call)
        assert call.op.name == ir.get_op("tile.reinterpret_view").name
        assert len(call.args) == 1
        assert call.kwargs == {"dtype": DataType.INT16}
        assert isinstance(call.type, ir.TileType)
        assert [dim.value for dim in call.type.shape if isinstance(dim, ir.ConstInt)] == [8, 32]

    def test_explicit_shape_wrapper(self):
        result = pl.tile.reinterpret_view(self._tile(), pl.INT16, shape=[4, 64])

        call = result.unwrap()
        assert isinstance(call, ir.Call)
        assert len(call.args) == 2
        shape_arg = call.args[1]
        assert isinstance(shape_arg, ir.MakeTuple)
        assert [dim.value for dim in shape_arg.elements if isinstance(dim, ir.ConstInt)] == [4, 64]
        assert isinstance(call.type, ir.TileType)
        assert [dim.value for dim in call.type.shape if isinstance(dim, ir.ConstInt)] == [4, 64]

    def test_exported_from_tile_namespace(self):
        assert "reinterpret_view" in pl.tile.__all__
        assert hasattr(pl.tile, "reinterpret_view")


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
        assert call.op.name == ir.get_op("tile.batch_matmul").name
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
        assert call.op.name == ir.get_op("tile.batch_matmul_acc").name
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
        assert call.op.name == _OP_TILE_TRANSPOSE
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
        assert call.op.name == ir.get_op("tile.row_max").name
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
        assert call.op.name == _OP_TILE_SLICE
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
                tmp: pl.Tile[[17, 32], pl.UINT8] = pl.tile.create(
                    [17, 32], dtype=pl.UINT8, target_memory=pl.MemorySpace.Vec
                )
                tile_c: pl.Tile[[16, 16], pl.FP32] = pl.prelu(tile_x, slope, tmp)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.prelu" in ir_str
        reparsed = pl.parse_program(ir_str)
        ir.assert_structural_equal(Program, reparsed)

    def test_tile_prelu_preserves_valid_shape(self):
        """TPRELU result mirrors the source physical and valid shapes."""
        src = _partial_tile([16, 16], [8, 12], name="src")
        slope = _partial_tile([16, 16], [8, 12], name="slope")
        span = ir.Span.unknown()
        tmp = ir.Var("tmp", ir.TileType([9, 32], DataType.UINT8), span)

        result = tile.prelu(src, slope, tmp).type

        assert isinstance(result, ir.TileType)
        assert [dim.value for dim in result.shape if isinstance(dim, ir.ConstInt)] == [16, 16]
        assert _valid_of(result) == [8, 12]

    def test_tile_prelu_defers_target_specific_tmp_validation(self):
        """IR deduction accepts a small UINT8 placeholder; A2/A3 validates it in codegen."""
        span = ir.Span.unknown()
        src = ir.Var("src", ir.TileType([16, 16], DataType.FP32), span)
        slope = ir.Var("slope", ir.TileType([16, 16], DataType.FP32), span)
        tmp = ir.Var("tmp", ir.TileType([1, 1], DataType.UINT8), span)

        result = tile.prelu(src, slope, tmp).type

        assert isinstance(result, ir.TileType)
        assert result.dtype == DataType.FP32

    def test_tile_prelu_defers_alias_validation_to_target_codegen(self):
        """Expression identity is not an alias proof, and A5 permits overlapping operands."""
        span = ir.Span.unknown()
        src = ir.Var("src", ir.TileType([16, 16], DataType.FP32), span)
        tmp = ir.Var("tmp", ir.TileType([17, 32], DataType.UINT8), span)

        result = tile.prelu(src, src, tmp)

        assert isinstance(result.type, ir.TileType)

    @pytest.mark.parametrize(
        "slope_type,error",
        [
            (ir.TileType([8, 16], DataType.FP32), "physical shape"),
            (ir.TileType([16, 16], DataType.FP16), "slope dtype"),
            (
                ir.TileType([16, 16], DataType.FP32, tile_view=ir.TileView(valid_shape=[8, 16])),
                "valid_shape",
            ),
        ],
    )
    def test_tile_prelu_rejects_incompatible_slope(self, slope_type, error):
        """TPRELU rejects slope contracts that PTOAS cannot assemble."""
        span = ir.Span.unknown()
        src = ir.Var("src", ir.TileType([16, 16], DataType.FP32), span)
        slope = ir.Var("slope", slope_type, span)
        tmp = ir.Var("tmp", ir.TileType([17, 32], DataType.UINT8), span)

        with pytest.raises(ValueError, match=error):
            tile.prelu(src, slope, tmp)

    def test_tile_prelu_rejects_non_rank2_tmp(self):
        """The target-independent ABI still requires a rank-2 tile placeholder."""
        span = ir.Span.unknown()
        src = ir.Var("src", ir.TileType([16, 16], DataType.FP32), span)
        slope = ir.Var("slope", ir.TileType([16, 16], DataType.FP32), span)
        tmp = ir.Var("tmp", ir.TileType([16], DataType.UINT8), span)

        with pytest.raises(ValueError, match="rank-2 tmp"):
            tile.prelu(src, slope, tmp)

    @pytest.mark.parametrize(
        ("src_type", "tmp_type", "error"),
        [
            (ir.TileType([16, 16], DataType.INT32), ir.TileType([17, 32], DataType.UINT8), "src dtype"),
            (ir.TileType([256], DataType.FP32), ir.TileType([17, 32], DataType.UINT8), "rank-2 src"),
        ],
    )
    def test_tile_prelu_rejects_invalid_src_contract(self, src_type, tmp_type, error):
        """TPRELU rejects unsupported source dtypes and ranks."""
        span = ir.Span.unknown()
        src = ir.Var("src", src_type, span)
        slope = ir.Var("slope", src_type, span)
        tmp = ir.Var("tmp", tmp_type, span)

        with pytest.raises(ValueError, match=error):
            tile.prelu(src, slope, tmp)

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
        """Test tile.sels operator - select between a tile and scalar via mask."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(a, [0, 0], [32, 32])
                mask: pl.Tile[[32, 32], pl.UINT8] = pl.cmps(tile_a, 0.0, cmp_type=4)
                tmp: pl.Tile[[1, 32], pl.UINT8] = pl.tile.create([1, 32], dtype=pl.UINT8)
                tile_out: pl.Tile[[32, 32], pl.FP32] = pl.sels(mask, tile_a, tmp, -1.0)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_out, [0, 0], output)
                return result

        ir_str = str(Program)
        assert "tile.sels" in ir_str
        reparsed = pl.parse_program(ir_str)
        ir.assert_structural_equal(Program, reparsed)

    def test_tile_sels_preserves_src_type_and_valid_shape(self):
        """TSELS result mirrors src rather than the packed mask or tmp."""
        span = ir.Span.unknown()
        mask = ir.Var(
            "mask",
            ir.TileType([16, 32], DataType.UINT8, tile_view=ir.TileView(valid_shape=[8, 2])),
            span,
        )
        src = _partial_tile([16, 16], [8, 12], name="src")
        tmp = ir.Var("tmp", ir.TileType([1, 32], DataType.UINT8), span)

        result = tile.sels(mask, src, tmp, -2.5).type

        assert isinstance(result, ir.TileType)
        assert result.dtype == DataType.FP32
        assert [dim.value for dim in result.shape if isinstance(dim, ir.ConstInt)] == [16, 16]
        assert _valid_of(result) == [8, 12]

    def test_tile_sels_retypes_constant_to_src_dtype(self):
        """A parser-produced constant adopts the selected source dtype."""
        span = ir.Span.unknown()
        mask = ir.Var("mask", ir.TileType([16, 32], DataType.UINT8), span)
        src = ir.Var("src", ir.TileType([16, 16], DataType.FP16), span)
        tmp = ir.Var("tmp", ir.TileType([1, 32], DataType.UINT8), span)
        scalar = ir.ConstFloat(-1.0, DataType.FP32, span)

        call = tile.sels(mask, src, tmp, scalar)

        assert _operand_dtype(call.args[3]) == DataType.FP16

    def test_tile_sels_rejects_fractional_constant_for_integer_src(self):
        """Retyping a scalar must not silently truncate a fractional value."""
        span = ir.Span.unknown()
        mask = ir.Var("mask", ir.TileType([16, 32], DataType.UINT8), span)
        src = ir.Var("src", ir.TileType([16, 16], DataType.INT32), span)
        tmp = ir.Var("tmp", ir.TileType([1, 32], DataType.UINT8), span)

        with pytest.raises(ValueError, match="non-integral"):
            tile.sels(mask, src, tmp, -1.5)

    def test_tile_sels_rejects_scalar_dtype_mismatch(self):
        """A non-constant scalar expression must match the selected source dtype."""
        span = ir.Span.unknown()
        mask = ir.Var("mask", ir.TileType([16, 32], DataType.UINT8), span)
        src = ir.Var("src", ir.TileType([16, 16], DataType.FP16), span)
        tmp = ir.Var("tmp", ir.TileType([1, 32], DataType.UINT8), span)
        scalar = ir.Var("scalar", ir.ScalarType(DataType.FP32), span)

        with pytest.raises(ValueError, match="scalar dtype"):
            tile.sels(mask, src, tmp, scalar)

    @pytest.mark.parametrize(
        "mask_type,error",
        [
            (ir.TileType([16, 32], DataType.FP32), "integer mask"),
            (ir.TileType([32], DataType.UINT8), "rank-2 mask"),
        ],
    )
    def test_tile_sels_rejects_invalid_mask(self, mask_type, error):
        """TSELS requires a rank-2 packed integer predicate tile."""
        span = ir.Span.unknown()
        mask = ir.Var("mask", mask_type, span)
        src = ir.Var("src", ir.TileType([16, 16], DataType.FP32), span)
        tmp = ir.Var("tmp", ir.TileType([1, 32], DataType.UINT8), span)

        with pytest.raises(ValueError, match=error):
            tile.sels(mask, src, tmp, -1.0)

    @pytest.mark.parametrize(
        "mask_type,error",
        [
            (
                ir.TileType([7, 64], DataType.UINT8),
                "mask carrier rows",
            ),
            (
                ir.TileType([8, 32], DataType.UINT8),
                "each mask carrier row",
            ),
        ],
    )
    def test_tile_sels_rejects_mask_too_small_for_src_valid_shape(self, mask_type, error):
        """A packed mask must cover every valid source row and column bit."""
        span = ir.Span.unknown()
        mask = ir.Var("mask", mask_type, span)
        src = ir.Var("src", ir.TileType([8, 257], DataType.FP32), span)
        tmp = ir.Var("tmp", ir.TileType([1, 32], DataType.UINT8), span)

        with pytest.raises(ValueError, match=error):
            tile.sels(mask, src, tmp, -1.0)

    def test_tile_sels_accepts_provable_dynamic_mask_coverage(self):
        """Shared symbolic rows and the exact packed-byte expression are provably safe."""
        span = ir.Span.unknown()
        valid_rows = ir.Var("valid_rows", ir.ScalarType(DataType.INDEX), span)
        valid_cols = ir.Var("valid_cols", ir.ScalarType(DataType.INDEX), span)
        packed_cols = (valid_cols + 7) // 8
        mask = ir.Var(
            "mask",
            ir.TileType(
                [16, 64],
                DataType.UINT8,
                tile_view=ir.TileView(valid_shape=[valid_rows, packed_cols]),
            ),
            span,
        )
        src = ir.Var(
            "src",
            ir.TileType(
                [16, 512],
                DataType.FP32,
                tile_view=ir.TileView(valid_shape=[valid_rows, valid_cols]),
            ),
            span,
        )
        tmp = ir.Var("tmp", ir.TileType([1, 32], DataType.UINT8), span)

        result = tile.sels(mask, src, tmp, -1.0)

        assert isinstance(result.type, ir.TileType)

    @pytest.mark.parametrize(
        ("mask_dtype", "physical_cols", "valid_cols", "accepted"),
        [
            (DataType.INT16, 32, 16, False),
            (DataType.INT16, 32, 17, True),
            (DataType.UINT16, 32, 16, False),
            (DataType.UINT16, 32, 17, True),
            (DataType.INT32, 16, 8, False),
            (DataType.INT32, 16, 9, True),
            (DataType.UINT32, 16, 8, False),
            (DataType.UINT32, 16, 9, True),
        ],
    )
    def test_tile_sels_packed_mask_capacity_respects_carrier_width(
        self, mask_dtype, physical_cols, valid_cols, accepted
    ):
        """Packed-mask capacity is measured in bytes for every integer carrier."""
        span = ir.Span.unknown()
        mask = ir.Var(
            "mask",
            ir.TileType(
                [2, physical_cols],
                mask_dtype,
                tile_view=ir.TileView(valid_shape=[2, valid_cols]),
            ),
            span,
        )
        src = ir.Var("src", ir.TileType([2, 257], DataType.FP32), span)
        tmp = ir.Var("tmp", ir.TileType([1, 32], DataType.UINT8), span)

        if accepted:
            assert isinstance(tile.sels(mask, src, tmp, -1.0).type, ir.TileType)
        else:
            with pytest.raises(ValueError, match="each mask carrier row"):
                tile.sels(mask, src, tmp, -1.0)

    @pytest.mark.parametrize(
        ("src_type", "tmp_type", "error"),
        [
            (ir.TileType([16, 16], DataType.BF16), ir.TileType([1, 32], DataType.UINT8), "src dtype"),
            (ir.TileType([256], DataType.FP32), ir.TileType([1, 32], DataType.UINT8), "rank-2 src"),
            (ir.TileType([16, 16], DataType.FP32), ir.TileType([32], DataType.UINT8), "rank-2 tmp"),
        ],
    )
    def test_tile_sels_rejects_invalid_src_and_tmp_contract(self, src_type, tmp_type, error):
        """TSELS rejects unsupported source dtypes and non-2D operands."""
        span = ir.Span.unknown()
        mask = ir.Var("mask", ir.TileType([16, 32], DataType.UINT8), span)
        src = ir.Var("src", src_type, span)
        tmp = ir.Var("tmp", tmp_type, span)

        with pytest.raises(ValueError, match=error):
            tile.sels(mask, src, tmp, -1.0)

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


class TestTileScalarOperandDtype:
    """A constant scalar operand of a tile x scalar op adopts the tile dtype.

    Regression: a DSL bare int literal is parsed to ``ConstInt(v, INDEX)``, and
    ``index`` is not a legal operand type for any ``pto.t*s`` instruction. The
    tile scalar wrappers must re-stamp such placeholders to the tile element
    dtype (``_normalize_scalar_operand``), reject non-constant ``index`` values,
    and leave already-typed / non-constant operands untouched.
    """

    # Every tile x scalar wrapper that normalizes a constant operand. Each entry
    # builds a 2-arg call ``fn(tile, 5)`` (extra positional args are appended).
    _INT_TILE_WRAPPERS = [
        ("adds", tile.adds, ()),
        ("subs", tile.subs, ()),
        ("muls", tile.muls, ()),
        ("divs", tile.divs, ()),
        ("maximums", tile.maximums, ()),
        ("minimums", tile.minimums, ()),
        ("cmps", tile.cmps, ()),
        ("ands", tile.ands, ()),
        ("ors", tile.ors, ()),
        ("shls", tile.shls, ()),
        ("shrs", tile.shrs, ()),
    ]

    @staticmethod
    def _int_tile(dtype=DataType.INT32, shape=(32, 32)):
        return ir.Var("t", ir.TileType(list(shape), dtype), ir.Span.unknown())

    def test_dsl_int_literal_adopts_tile_dtype(self):
        """`pl.add(tile_i32, 5)` yields an INT32 scalar operand, not index."""
        for dtype in (DataType.INT32, DataType.INT16, DataType.INT8):
            for fn in (tile.add, tile.sub, tile.mul):
                call = fn(self._int_tile(dtype), 5)
                rhs = call.args[1]
                assert isinstance(rhs, ir.ConstInt)
                assert rhs.dtype == dtype, f"{fn.__name__} on {dtype}: {rhs.dtype}"

    def test_no_index_survives_any_tile_scalar_op(self):
        """The direct regression: no wrapper leaves an index-typed constant."""
        for name, fn, extra in self._INT_TILE_WRAPPERS:
            call = fn(self._int_tile(DataType.INT32), 5, *extra)
            dtype = _operand_dtype(call.args[1])
            assert dtype != DataType.INDEX, f"{name} left an index operand"
            assert dtype == DataType.INT32, f"{name}: {dtype}"

    def test_int_literal_on_float_tile_becomes_const_float(self):
        """An int literal on a float tile becomes a ConstFloat at the tile dtype.

        Codegen dispatches on the node kind, so an int operand on a float tile
        must be a ConstFloat or MLIR receives ``arith.constant 5 : f32``.
        """
        for dtype in (DataType.FP16, DataType.FP32, DataType.BF16):
            call = tile.adds(ir.Var("t", ir.TileType([32, 32], dtype), ir.Span.unknown()), 5)
            rhs = call.args[1]
            assert isinstance(rhs, ir.ConstFloat)
            assert rhs.dtype == dtype

    def test_float_literal_on_int_tile_keeps_fp32(self):
        """A float literal on an int tile keeps FP32 (promotion is preserved)."""
        call = tile.adds(self._int_tile(DataType.INT32), 2.5)
        rhs = call.args[1]
        assert isinstance(rhs, ir.ConstFloat)
        assert rhs.dtype == DataType.FP32
        # The tile op still deduces its result at the tile dtype.
        assert _tile_result_dtype(call) == DataType.INT32

    def test_subs_mixed_scalar_dtype_preserves_tile_dtype(self):
        """tsubs accepts a wider scalar without retyping its tile result."""
        span = ir.Span.unknown()
        lhs = self._int_tile(DataType.INT16)
        scalar = ir.ConstFloat(2.5, DataType.FP32, span)

        explicit_call = tile.subs(lhs, scalar)
        literal_call = tile.subs(lhs, 2.5)

        assert explicit_call.args[1] is scalar
        assert _operand_dtype(explicit_call.args[1]) == DataType.FP32
        assert _operand_dtype(literal_call.args[1]) == DataType.FP32
        assert _tile_result_dtype(explicit_call) == DataType.INT16
        assert _tile_result_dtype(literal_call) == DataType.INT16

    def test_subs_rejects_unsupported_tile_dtype(self):
        """INT64 is outside the current pto.tsubs tile dtype union."""
        lhs = self._int_tile(DataType.INT64)

        with pytest.raises(ValueError, match=r"INT8, INT16, INT32, FP16, FP32, BF16"):
            tile.subs(lhs, 1)

    @pytest.mark.parametrize(
        "dtype",
        [DataType.UINT32, DataType.BOOL, DataType.INDEX, DataType.INT64, DataType.FP8E4M3FN],
    )
    def test_subs_rejects_unsupported_scalar_dtype(self, dtype):
        """Only scalar dtypes exercised by the executable PTOAS paths are exposed."""
        span = ir.Span.unknown()
        lhs = self._int_tile(DataType.INT16)
        scalar = ir.Var("scalar", ir.ScalarType(dtype), span)

        with pytest.raises(ValueError, match=r"requires scalar dtype in"):
            ir.create_op_call("tile.subs", [lhs, scalar], span)

    def test_float_literal_on_float_tile_adopts_tile_dtype(self):
        """A float literal on a low-precision float tile adopts the tile dtype.

        `tile.adds(fp16_tile, 2.5)` -> ConstFloat(fp16) (previously fp32); the
        scalar follows the tile element dtype rather than defaulting to FP32.
        """
        for dtype in (DataType.FP16, DataType.BF16):
            call = tile.adds(ir.Var("t", ir.TileType([32, 32], dtype), ir.Span.unknown()), 2.5)
            rhs = call.args[1]
            assert isinstance(rhs, ir.ConstFloat)
            assert rhs.dtype == dtype

    def test_bitwise_literal_adopts_tile_dtype(self):
        """Bitwise scalar ops re-stamp the literal to the tile dtype."""
        for fn in (tile.ands, tile.ors):
            call = fn(self._int_tile(DataType.INT16), 255)
            assert _operand_dtype(call.args[1]) == DataType.INT16

    def test_narrow_shift_literal_adopts_tile_dtype(self):
        """Shift counts follow the tile dtype, including narrow int tiles.

        `DeduceTileOpIntScalarBinaryType` permits any integer width for the
        shift operand ("codegen casts to i32"), and an i8/i16 shift count is
        accepted end-to-end by ptoas, so narrowing here is safe.
        """
        for dtype in (DataType.INT8, DataType.INT16):
            for fn in (tile.shls, tile.shrs):
                call = fn(self._int_tile(dtype), 3)
                assert _operand_dtype(call.args[1]) == dtype, f"{fn.__name__} on {dtype}"

    def test_ir_level_restamps_index_const(self):
        """A hand-built ConstInt(INDEX) operand (parser output) is re-stamped."""
        span = ir.Span.unknown()
        call = tile.adds(self._int_tile(DataType.INT16), ir.ConstInt(5, DataType.INDEX, span))
        assert _operand_dtype(call.args[1]) == DataType.INT16

    def test_explicitly_typed_const_is_not_restamped(self):
        """An explicit pl.const(v, dtype) is a user annotation, left untouched."""
        span = ir.Span.unknown()
        typed = ir.ConstInt(5, DataType.INT32, span)
        call = tile.adds(ir.Var("t", ir.TileType([32, 32], DataType.INT16), span), typed)
        # INT32 constant preserved even though the tile is INT16.
        assert _operand_dtype(call.args[1]) == DataType.INT32

    def test_index_scalar_value_is_rejected(self):
        """A non-constant index scalar (loop var, dim) is rejected with a hint."""
        span = ir.Span.unknown()
        idx = ir.Var("i", ir.ScalarType(DataType.INDEX), span)
        with pytest.raises((ValueError, TypeError), match="index"):
            tile.adds(self._int_tile(DataType.INT32), idx)

    def test_index_scalar_reject_hint_names_cast(self):
        """The rejection points the user at pl.cast."""
        span = ir.Span.unknown()
        idx = ir.Var("i", ir.ScalarType(DataType.INDEX), span)
        with pytest.raises((ValueError, TypeError), match="pl.cast"):
            tile.adds(self._int_tile(DataType.INT32), idx)

    def test_typed_scalar_param_passes_through(self):
        """A typed pl.Scalar operand is not re-stamped."""
        span = ir.Span.unknown()
        k = ir.Var("k", ir.ScalarType(DataType.INT32), span)
        call = tile.adds(ir.Var("t", ir.TileType([32, 32], DataType.INT16), span), k)
        assert call.args[1] is k

    def test_tile_rhs_untouched(self):
        """A tile rhs dispatches to the tile-tile op, operand unchanged."""
        span = ir.Span.unknown()
        lhs = ir.Var("a", ir.TileType([32, 32], DataType.INT32), span)
        rhs = ir.Var("b", ir.TileType([32, 32], DataType.INT32), span)
        call = tile.add(lhs, rhs)
        assert call.op.name == ir.get_op("tile.add").name
        assert call.args[1] is rhs

    def test_expands_literal_adopts_target_dtype(self):
        """tile.expands re-stamps its scalar to the target tile dtype (not FP32)."""
        call = tile.expands(self._int_tile(DataType.INT32), 5)
        assert _operand_dtype(call.args[1]) == DataType.INT32

    def test_lrelu_slope_stays_fp32(self):
        """tile.lrelu keeps its slope at FP32 and never leaves it index."""
        call = tile.lrelu(ir.Var("t", ir.TileType([32, 32], DataType.FP32), ir.Span.unknown()), 1)
        assert _operand_dtype(call.args[1]) == DataType.FP32

    @pytest.mark.parametrize(
        "dtype,scalar,expected_dtype,expected_value",
        [
            (DataType.INT8, -2, DataType.INT8, -2),
            (DataType.UINT8, 0x82, DataType.INT8, -126),
            (DataType.INT16, -3, DataType.INT16, -3),
            (DataType.UINT16, 0x8007, DataType.INT16, -32761),
            (DataType.INT32, 7, DataType.INT32, 7),
            (DataType.UINT32, 0x8000000B, DataType.INT32, -2147483637),
            (DataType.FP16, -0.5, DataType.FP16, -0.5),
            (DataType.FP32, 1.25, DataType.FP32, 1.25),
        ],
    )
    def test_sels_scalar_adopts_ptoas_dtype(self, dtype, scalar, expected_dtype, expected_value):
        """tile.sels uses signed bit-compatible scalars for unsigned sources."""
        span = ir.Span.unknown()
        mask = ir.Var("mask", ir.TileType([32, 32], DataType.UINT8), span)
        src = ir.Var("src", ir.TileType([32, 32], dtype), span)
        tmp = ir.Var("tmp", ir.TileType([1, 32], DataType.UINT8), span)
        call = tile.sels(mask, src, tmp, scalar)
        scalar_arg = call.args[3]
        assert isinstance(scalar_arg, (ir.ConstInt, ir.ConstFloat))
        assert _operand_dtype(scalar_arg) == expected_dtype
        assert scalar_arg.value == expected_value

    def test_sels_unsigned_src_accepts_only_signed_same_width_scalar_expr(self):
        """PTOAS scalar operands are signed even when the selected tile is unsigned."""
        span = ir.Span.unknown()
        mask = ir.Var("mask", ir.TileType([32, 32], DataType.UINT8), span)
        src = ir.Var("src", ir.TileType([32, 32], DataType.UINT16), span)
        tmp = ir.Var("tmp", ir.TileType([1, 32], DataType.UINT8), span)

        call = tile.sels(mask, src, tmp, ir.Var("signed_scalar", ir.ScalarType(DataType.INT16), span))
        assert isinstance(call.type, ir.TileType)
        assert call.type.dtype == DataType.UINT16

        with pytest.raises(ValueError, match="requires scalar dtype int16 for src dtype uint16"):
            tile.sels(mask, src, tmp, ir.Var("unsigned_scalar", ir.ScalarType(DataType.UINT16), span))


class TestTileLoadOp:
    """Tests for tile.load operation with valid_shape and TileView."""

    def test_load_uses_singular_valid_shape_keyword(self):
        params = inspect.signature(tile.load).parameters
        removed_plural = "valid_" + "shapes"
        assert "valid_shape" in params
        assert removed_plural not in params

    def test_load_rejects_removed_plural_keyword(self):
        span = ir.Span.unknown()
        tensor = ir.Var("a", ir.TensorType([64, 128], DataType.FP32), span)
        removed_plural = "valid_" + "shapes"
        with pytest.raises(TypeError, match=rf"unexpected keyword argument '{removed_plural}'"):
            cast(Any, tile.load)(tensor, [0, 0], [64, 128], **{removed_plural: [32, 128]})

    def test_load_without_valid_shape_sets_tileview_from_shapes(self):
        """When valid_shape is not provided, TileView.valid_shape equals shapes."""
        span = ir.Span.unknown()
        dim64 = ir.ConstInt(64, DataType.INT32, span)
        dim128 = ir.ConstInt(128, DataType.INT32, span)
        tensor_type = ir.TensorType([dim64, dim128], DataType.FP32)
        tensor = ir.Var("a", tensor_type, span)

        call = tile.load(tensor, [0, 0], [64, 128])
        tile_type = call.type

        assert isinstance(tile_type, ir.TileType)
        assert len(tile_type.get_effective_tile_view().valid_shape) == 2

    def test_load_with_static_valid_shape_sets_tileview(self):
        """When valid_shape is provided as static ints, TileView.valid_shape reflects it."""
        span = ir.Span.unknown()
        dim64 = ir.ConstInt(64, DataType.INT32, span)
        dim128 = ir.ConstInt(128, DataType.INT32, span)
        tensor_type = ir.TensorType([dim64, dim128], DataType.FP32)
        tensor = ir.Var("a", tensor_type, span)

        call = tile.load(tensor, [0, 0], [128, 128], valid_shape=[64, 128])
        tile_type = call.type

        assert isinstance(tile_type, ir.TileType)
        assert tile_type.tile_view is not None
        assert len(tile_type.tile_view.valid_shape) == 2
        # tile shape should still be [128, 128]
        assert len(tile_type.shape) == 2

    def test_load_with_dynamic_valid_shape_sets_tileview(self):
        """When valid_shape is provided as symbolic vars, TileView.valid_shape uses them."""
        span = ir.Span.unknown()
        dim64 = ir.ConstInt(64, DataType.INT32, span)
        dim128 = ir.ConstInt(128, DataType.INT32, span)
        tensor_type = ir.TensorType([dim64, dim128], DataType.FP32)
        tensor = ir.Var("a", tensor_type, span)
        M = ir.Var("M", ir.ScalarType(DataType.INT64), span)
        N = ir.Var("N", ir.ScalarType(DataType.INT64), span)

        call = tile.load(tensor, [0, 0], [64, 128], valid_shape=[M, N])
        tile_type = call.type

        assert isinstance(tile_type, ir.TileType)
        assert tile_type.tile_view is not None
        assert len(tile_type.tile_view.valid_shape) == 2
        # valid_shape elements should be the symbolic vars M and N
        assert tile_type.tile_view.valid_shape[0] is M
        assert tile_type.tile_view.valid_shape[1] is N

    def test_load_via_pl_load_with_valid_shape(self):
        """pl.load with valid_shape propagates TileView to the output tile."""

        @pl.program
        class Prog:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                M: pl.Scalar[pl.INT64],
                N: pl.Scalar[pl.INT64],
            ) -> pl.Tile[[128, 128], pl.FP32]:
                tile: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [0, 0], [128, 128], valid_shape=[M, N])
                return tile

        # Just verifying it builds without error
        assert Prog is not None


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


class TestTileMoveOp:
    """Tests for tile.move result-view deduction."""

    @staticmethod
    def _tile_var(space):
        return ir.Var("t_src", ir.TileType([16, 64], DataType.FP32, None, None, space), ir.Span.unknown())

    @pytest.mark.parametrize(
        ("space", "expected_fractal"),
        [
            (ir.MemorySpace.Vec, 512),
            (ir.MemorySpace.Mat, 512),
            (ir.MemorySpace.Acc, 1024),
        ],
    )
    def test_same_space_move_deduces_destination_implicit_view(self, space, expected_fractal):
        """`fractal` is the destination buffer's boxing granularity, not the
        TileView default: Acc (L0C) is NZ-boxed at 1024. A move that changes
        neither space nor layout must therefore deduce exactly the destination's
        implicit view, which is stored canonically as None. With the default 512
        an Acc result stays explicit — and since a pass-synthesized move's LHS
        Var carries no view, the print->parse roundtrip breaks."""
        call = tile.move(self._tile_var(space), target_memory=space)

        assert isinstance(call.type, ir.TileType)
        assert call.type.memory_space == space
        assert call.type.tile_view is None
        assert call.type.get_effective_tile_view().fractal == expected_fractal

    @pytest.mark.parametrize("source_space", [ir.MemorySpace.Mat, ir.MemorySpace.Vec])
    def test_move_into_acc_adopts_acc_layout_from_non_acc_source(self, source_space):
        """A 512-fractal source moved into Acc must report Acc's full NZ view.

        The same-space cases above cannot show this: an Acc source already
        carries Acc's layout, so they pass even if the result wrongly inherited
        it from the source instead of taking it from the destination.

        Acc dictates blayout/slayout as well as fractal — a tile living in L0C is
        NZ-boxed whatever it was moved from — so the deduced view is exactly
        Acc's implicit view and is stored canonically as None, for any source.
        """
        source = self._tile_var(source_space)
        source_type = source.type
        assert isinstance(source_type, ir.TileType)
        assert source_type.get_effective_tile_view().fractal == 512

        call = tile.move(source, target_memory=ir.MemorySpace.Acc)

        assert isinstance(call.type, ir.TileType)
        assert call.type.memory_space == ir.MemorySpace.Acc
        assert call.type.tile_view is None
        eff = call.type.get_effective_tile_view()
        assert (eff.blayout, eff.slayout, eff.fractal) == (
            ir.TileLayout.col_major,
            ir.TileLayout.row_major,
            1024,
        )

    def test_acc_to_vec_move_keeps_vec_fractal(self):
        """Moving out of Acc must adopt Vec's granularity, not carry 1024 along:
        the cube->vec pipe un-fractalizes the data during transfer."""
        call = tile.move(
            self._tile_var(ir.MemorySpace.Acc),
            target_memory=ir.MemorySpace.Vec,
            blayout=ir.TileLayout.row_major,
            slayout=ir.TileLayout.none_box,
        )

        assert isinstance(call.type, ir.TileType)
        assert call.type.get_effective_tile_view().fractal == 512
        # row_major/none_box/512 *is* Vec's implicit view — stored canonically.
        assert call.type.tile_view is None


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
        assert call.op.name == ir.get_op("tile.assemble").name
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
        assert call.op.name == _OP_TILE_EXTRACT
        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        assert result_type.dtype == DataType.FP16
        assert len(result_type.shape) == 2
        rows, cols = result_type.shape
        assert isinstance(rows, ir.ConstInt) and rows.value == 64
        assert isinstance(cols, ir.ConstInt) and cols.value == 64
        assert result_type.get_effective_tile_view().compact == ir.CompactMode.null

    @pytest.mark.parametrize(
        ("source_shape", "source_valid", "extract_shape", "target_memory", "expected_valid"),
        [
            ((384, 32), (384, 16), (192, 32), ir.MemorySpace.Right, (192, 16)),
            ((32, 384), (16, 384), (32, 192), ir.MemorySpace.Left, (16, 192)),
        ],
        ids=["right-n-tail", "left-m-tail"],
    )
    def test_tile_extract_partial_l0_operand_infers_compact_mode(
        self, source_shape, source_valid, extract_shape, target_memory, expected_valid
    ):
        """Partial Mat->L0 extracts select the valid-aware compact transfer."""
        span = ir.Span.unknown()
        src_type = ir.TileType(
            source_shape,
            DataType.INT8,
            tile_view=ir.TileView(valid_shape=source_valid),
            memory_space=ir.MemorySpace.Mat,
        )
        src = ir.Var("src", src_type, span)

        call = tile.extract(src, 0, 0, shape=extract_shape, target_memory=target_memory)

        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        assert result_type.memory_space == target_memory
        assert _valid_of(result_type) == list(expected_valid)
        assert result_type.get_effective_tile_view().compact == ir.CompactMode.normal

    def test_tile_extract_partial_non_l0_destination_stays_noncompact(self):
        """Compact inference is specific to boxed L0 operand transfers."""
        span = ir.Span.unknown()
        src = ir.Var(
            "src",
            ir.TileType(
                [64, 64],
                DataType.FP32,
                tile_view=ir.TileView(valid_shape=[64, 32]),
                memory_space=ir.MemorySpace.Acc,
            ),
            span,
        )

        result_type = tile.extract(src, 0, 0, shape=[32, 64], target_memory=ir.MemorySpace.Mat).type

        assert isinstance(result_type, ir.TileType)
        assert _valid_of(result_type) == [32, 32]
        assert result_type.get_effective_tile_view().compact == ir.CompactMode.null

    def test_tile_extract_acc_to_mat(self):
        """Acc source → Mat target: src lives in Acc, dtype preserved."""
        src_var = self._make_src_var(64, 64, DataType.FP32, memory_space=ir.MemorySpace.Acc)
        src_tile_type = src_var.type
        assert isinstance(src_tile_type, ir.TileType)
        assert src_tile_type.memory_space == ir.MemorySpace.Acc

        call = tile.extract(src_var, 0, 0, shape=[32, 32], target_memory=ir.MemorySpace.Mat)

        assert call.op.name == _OP_TILE_EXTRACT
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

        assert call.op.name == _OP_TILE_EXTRACT
        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        rows, cols = result_type.shape
        assert isinstance(rows, ir.ConstInt) and rows.value == 16
        assert isinstance(cols, ir.ConstInt) and cols.value == 16

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
        assert call.op.name == ir.get_op("tile.scatter_update").name
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
        assert call.op.name == _OP_TILE_MSCATTER
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
        assert call.op.name == _OP_TILE_MSCATTER
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
        assert call.op.name == ir.get_op("tile.scatter").name
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
        assert call.op.name == ir.get_op("tile.scatter_mask").name
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
        assert call.op.name == ir.get_op("tile.concat").name
        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        assert result_type.dtype == DataType.FP32
        assert len(result_type.shape) == 2
        # Output cols = 16 + 16 = 32
        assert isinstance(result_type.shape[1], ir.ConstInt)
        assert result_type.shape[1].value == 32

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

        with pytest.raises(InvalidOperationError, match="requires third argument to be a TensorType"):

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

        with pytest.raises(InvalidOperationError, match="requires first argument to be a TensorType"):

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


class TestWindowReadValidRegion:
    """The valid-region rule shared by tile.load, tile.slice and tile.extract.

    available    = clamp(source_valid - offset, 0, window)
    result_valid = min(requested_valid, available)
    """

    @staticmethod
    def _partial_tensor(shape, valid_shape, name="a"):
        span = ir.Span.unknown()
        view = ir.TensorView(stride=[], layout=ir.TensorLayout.ND, valid_shape=valid_shape)
        return ir.Var(name, ir.TensorType(shape, DataType.FP32, tensor_view=view), span)

    # --- tile.slice ---------------------------------------------------------

    def test_slice_full_source_stays_fully_valid(self):
        """A window inside a fully-valid source needs no valid_shape at all."""
        span = ir.Span.unknown()
        src = ir.Var("src", ir.TileType([64, 64], DataType.FP32), span)

        call = tile.slice(src, [16, 32], [8, 0])

        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        assert result_type.tile_view is None

    def test_slice_partial_source_narrows_result(self):
        """A window over padding inherits the source tile's narrower validity."""
        src = _partial_tile([64, 64], [40, 50])

        call = tile.slice(src, [32, 32], [24, 32])

        # rows: clamp(40 - 24, 0, 32) = 16;  cols: clamp(50 - 32, 0, 32) = 18
        assert _valid_of(call.type) == [16, 18]

    def test_slice_intersects_rather_than_replaces_explicit_valid_shape(self):
        """An explicit valid_shape narrows the result but cannot widen it."""
        src = _partial_tile([64, 64], [20, 64])

        widening = tile.slice(src, [32, 32], [0, 0], valid_shape=[32, 32])
        assert _valid_of(widening.type) == [20, 32]

        narrowing = tile.slice(src, [32, 32], [0, 0], valid_shape=[8, 4])
        assert _valid_of(narrowing.type) == [8, 4]

    def test_slice_folds_constants_without_min_max_nesting(self):
        """Static intersections fold to a plain ConstInt, not a min/max tree."""
        src = _partial_tile([64, 64], [40, 64])

        call = tile.slice(src, [32, 64], [16, 0], valid_shape=[32, 64])

        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        assert result_type.tile_view is not None
        # clamp(40 - 16, 0, 32) = 24, intersected with the request 32 -> 24.
        rows = result_type.tile_view.valid_shape[0]
        assert isinstance(rows, ir.ConstInt)
        assert rows.value == 24

    def test_slice_rejects_static_out_of_bounds_window(self):
        """A non-clamping slice that provably reads past the source is rejected."""
        span = ir.Span.unknown()
        src = ir.Var("src", ir.TileType([64, 64], DataType.FP32), span)

        with pytest.raises(ValueError, match="reads past the end of dimension 0"):
            tile.slice(src, [32, 64], [48, 0])

    def test_slice_rejects_negative_offset(self):
        """A provably negative offset starts outside the source."""
        span = ir.Span.unknown()
        src = ir.Var("src", ir.TileType([64, 64], DataType.FP32), span)
        neg = ir.ConstInt(-8, DataType.INDEX, span)

        with pytest.raises(ValueError, match="provably negative"):
            tile.slice(src, [16, 64], [neg, 0])

    def test_slice_has_no_clamp_escape_hatch(self):
        """An on-chip window cannot be clamped, so an overhang stays an error.

        `pto.subview` is a pure view and the Mat/Vec fold in CanonicalizeTileSlice
        turns the window into an ISA TEXTRACT, whose bounds are hard. Nothing can
        clamp a tile window, so `pl.slice` says so instead of offering a flag it
        cannot honour. The tensor boundary is where a ragged read gets clamped.
        """
        span = ir.Span.unknown()
        src = ir.Var("src", ir.TileType([96, 64], DataType.FP32), span)

        with pytest.raises(ValueError, match="no clamping mechanism"):
            tile.slice(src, [64, 64], [64, 0])

        # And the DSL rejects the flag itself rather than silently dropping it.
        tile_arg = pl.Tile(expr=src)
        with pytest.raises(TypeError, match="clamp=True is not supported for a Tile"):
            pl.slice(tile_arg, [64, 64], [64, 0], clamp=True)

    def test_slice_drop_dims_rejected_when_axis_is_not_provably_valid(self):
        """Rank reduction erases an axis, so the axis must have nothing left to say."""
        src = _partial_tile([64, 64], [8, 64])

        with pytest.raises(ValueError, match="not provably 1"):
            tile.slice(src, [1, 64], [16, 0], drop_dims=[0])

    def test_slice_inherits_source_pad_mode(self):
        """A read view over padded bytes keeps saying they are padded."""
        src = _partial_tile([64, 64], [40, 64], pad=ir.PadValue.zero)

        call = tile.slice(src, [32, 64], [0, 0])

        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        assert result_type.tile_view is not None
        assert result_type.tile_view.pad == ir.PadValue.zero

    # --- tile.load ----------------------------------------------------------

    def test_load_partial_source_narrows_the_tile(self):
        """A load can never report source padding as real data."""
        src = self._partial_tensor([64, 128], [40, 128])

        call = tile.load(src, [0, 0], [64, 128], valid_shape=[64, 128])

        # The request asked for all 64 rows; only 40 exist.
        assert _valid_of(call.type) == [40, 128]

    def test_load_rejects_a_request_that_reads_past_the_source(self):
        """valid_shape is what the DMA actually reads, so it must exist."""
        span = ir.Span.unknown()
        tensor_var = ir.Var("a", ir.TensorType([100, 128], DataType.FP32), span)

        # Claiming 64 valid rows at offset 64 reads to row 128 of a 100-row tensor.
        with pytest.raises(ValueError, match="reads past the end of dimension 0"):
            tile.load(tensor_var, [64, 0], [64, 128], valid_shape=[64, 128])

    def test_load_tile_may_overhang_the_source(self):
        """The destination tile is an allocation, so only the read extent must fit."""
        span = ir.Span.unknown()
        tensor_var = ir.Var("a", ir.TensorType([100, 128], DataType.FP32), span)

        # A 64-row tile at offset 64 overhangs, but only 36 rows are read.
        call = tile.load(tensor_var, [64, 0], [64, 128], valid_shape=[36, 128])

        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        assert [d.value for d in result_type.shape if isinstance(d, ir.ConstInt)] == [64, 128]
        assert _valid_of(result_type) == [36, 128]

    def test_load_clamp_narrows_an_over_reaching_request(self):
        """clamp=True cuts an over-reaching read back to the source edge."""
        span = ir.Span.unknown()
        tensor_var = ir.Var("a", ir.TensorType([100, 128], DataType.FP32), span)

        call = tile.load(tensor_var, [64, 0], [64, 128], valid_shape=[64, 128], clamp=True)

        # clamp(100 - 64, 0, 64) = 36, intersected with the 64-row request -> 36.
        assert _valid_of(call.type) == [36, 128]

    def test_load_clamp_print_parse_roundtrip(self):
        """A clamped ragged load survives python_print -> pl.parse -> python_print."""
        src = (
            "import pypto.language as pl\n\n"
            "@pl.program\n"
            "class P:\n"
            "    @pl.function\n"
            "    def main(self, x: pl.Tensor[[100, 128], pl.FP32]) -> pl.Tile[[64, 128], pl.FP32]:\n"
            "        t: pl.Tile[[64, 128], pl.FP32] = "
            "pl.tile.load(x, [64, 0], [64, 128], [64, 128], clamp=True)\n"
            "        return t\n"
        )
        prog = pl.parse(src)
        reparsed = pl.parse(ir.python_print(prog))
        ir.assert_structural_equal(reparsed, prog)

    def test_load_lower_rank_window_keeps_its_valid_shape(self):
        """A 2D tile out of a 3D tensor is a reinterpreting read, not a rectangle."""
        span = ir.Span.unknown()
        tensor_var = ir.Var("a", ir.TensorType([4, 128, 64], DataType.FP32), span)

        # Window rank 2 over a rank-3 source: the rule does not apply, so the
        # requested valid_shape passes through untouched rather than being
        # intersected against the wrong axes.
        call = tile.load(tensor_var, [0, 0], [16, 64], valid_shape=[16, 64])

        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        assert [d.value for d in result_type.shape if isinstance(d, ir.ConstInt)] == [16, 64]

    def test_load_rejects_a_request_larger_than_the_tile_that_holds_it(self):
        """valid <= shape is the standing invariant of the type the read produces.

        The source is big enough that the read stays in bounds, so only the window
        itself catches this: asking for 128 valid rows of a 64-row tile would put a
        valid region into the result that is larger than the shape holding it.
        """
        span = ir.Span.unknown()
        tensor_var = ir.Var("a", ir.TensorType([256, 128], DataType.FP32), span)

        with pytest.raises(ValueError, match="exceeds the window extent"):
            tile.load(tensor_var, [0, 0], [64, 128], valid_shape=[128, 128])

    def test_load_keeps_the_request_when_the_source_extent_is_undecidable(self):
        """An undecidable source extent is trusted, not folded into a runtime min.

        A source valid extent lives in the *type*, and may name a symbol that has no
        value in the reading function: a `pl.dynamic()` dim in a parameter's
        valid_shape is bound at the call site, so a standalone (precompiled) kernel
        never receives it. Folding it into a min would emit an operand that does not
        exist. Since the relation to the request cannot be decided either way, the
        request -- the only extent the operator can name -- stands.
        """
        span = ir.Span.unknown()
        # `SRC_VALID` stands for the type-level symbol; `valid_len` for the value the
        # kernel is actually handed. Nothing relates them.
        src_valid = ir.Var("SRC_VALID", ir.ScalarType(DataType.INDEX), span)
        valid_len = ir.Var("valid_len", ir.ScalarType(DataType.INDEX), span)
        view = ir.TensorView(stride=[], layout=ir.TensorLayout.ND, valid_shape=[16, src_valid])
        tensor_var = ir.Var("a", ir.TensorType([16, 128], DataType.FP32, tensor_view=view), span)

        call = tile.load(tensor_var, [0, 0], [16, 128], valid_shape=[16, valid_len])

        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        assert result_type.tile_view is not None
        # The request survives verbatim -- no min() wrapped around SRC_VALID.
        assert result_type.tile_view.valid_shape[1] is valid_len

    def test_load_symbolic_valid_shape_survives_unchanged(self):
        """A symbolic request is trusted: it is the caller's contract, not a guess."""
        span = ir.Span.unknown()
        tensor_var = ir.Var("a", ir.TensorType([64, 128], DataType.FP32), span)
        m = ir.Var("M", ir.ScalarType(DataType.INT64), span)

        call = tile.load(tensor_var, [0, 0], [64, 128], valid_shape=[m, 128])

        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        assert result_type.tile_view is not None
        # No redundant min() wrapped around the request.
        assert result_type.tile_view.valid_shape[0] is m

    # --- tile.extract -------------------------------------------------------

    def test_extract_full_source_stays_fully_valid(self):
        """An extract out of a fully-valid source needs no valid_shape."""
        span = ir.Span.unknown()
        src = ir.Var("src", ir.TileType([64, 256], DataType.FP16), span)

        call = tile.extract(src, 0, 0, shape=[64, 64], target_memory=ir.MemorySpace.Left)

        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        eff = result_type.get_effective_tile_view()
        assert [d.value for d in eff.valid_shape if isinstance(d, ir.ConstInt)] == [64, 64]

    def test_extract_partial_source_narrows_result(self):
        """TEXTRACT repacks a window, so it can only be valid where src is."""
        src = _partial_tile([64, 256], [40, 100])

        call = tile.extract(src, 16, 64, shape=[32, 32], target_memory=ir.MemorySpace.Vec)

        # rows: clamp(40 - 16, 0, 32) = 24;  cols: clamp(100 - 64, 0, 32) = 32
        assert _valid_of(call.type) == [24, 32]


class TestDestinationSpaceLayoutDeduction:
    """A retargeting op's result layout comes from the DESTINATION memory space.

    tile.move / tile.extract used to deduce the result view against a nullopt
    memory space and let OpRegistry::Create stamp the real space by rebuilding
    the TileType -- which re-ran CanonicalizeTileViewInPlace against a different
    implicit view. Since the collapse to nullopt only fires when the view is
    fully valid and unpadded, the destination layout silently depended on the
    tile's valid extent: a fully-valid Vec->Mat move got Mat's boxed NZ layout,
    while the same move on a ragged tile kept Vec's flat row_major/none_box.

    The layout must depend only on (shape, destination space).
    """

    # Destination space -> the (blayout, slayout, fractal) a tile living there
    # must carry. Acc is the one destination with a non-default fractal, so it
    # also pins that fractal comes from the destination and is never inherited.
    _BOXED_DESTINATIONS = [
        (ir.MemorySpace.Mat, ir.TileLayout.col_major, ir.TileLayout.row_major, 512),
        (ir.MemorySpace.Left, ir.TileLayout.col_major, ir.TileLayout.row_major, 512),
        (ir.MemorySpace.Right, ir.TileLayout.row_major, ir.TileLayout.col_major, 512),
        (ir.MemorySpace.Acc, ir.TileLayout.col_major, ir.TileLayout.row_major, 1024),
    ]

    @staticmethod
    def _layout_of(result_type):
        eff = result_type.get_effective_tile_view()
        return (eff.blayout, eff.slayout, eff.fractal)

    # --- tile.move ----------------------------------------------------------

    @pytest.mark.parametrize("space,blayout,slayout,fractal", _BOXED_DESTINATIONS)
    def test_move_layout_is_independent_of_the_source_valid_extent(self, space, blayout, slayout, fractal):
        """A narrower valid_shape must not change where the result's layout comes from."""
        span = ir.Span.unknown()
        full = ir.Var("full", ir.TileType([64, 64], DataType.FP32), span)
        ragged = _partial_tile([64, 64], [64, 48])

        full_layout = self._layout_of(tile.move(full, target_memory=space).type)
        ragged_layout = self._layout_of(tile.move(ragged, target_memory=space).type)

        assert full_layout == ragged_layout == (blayout, slayout, fractal)

    @pytest.mark.parametrize("space,blayout,slayout,fractal", _BOXED_DESTINATIONS)
    def test_move_layout_is_independent_of_the_source_pad(self, space, blayout, slayout, fractal):
        """pad is the other reason a view cannot collapse -- same rule applies."""
        padded = _partial_tile([64, 64], [64, 64], pad=ir.PadValue.zero)

        assert self._layout_of(tile.move(padded, target_memory=space).type) == (blayout, slayout, fractal)

    def test_move_narrowing_the_source_still_narrows_the_result(self):
        """Fixing the layout must not drop the valid extent the move carries over."""
        ragged = _partial_tile([64, 64], [64, 48])

        assert _valid_of(tile.move(ragged, target_memory=ir.MemorySpace.Mat).type) == [64, 48]

    def test_move_to_right_keeps_row_major_for_a_column_vector(self):
        """L0B needs a RowMajor block layout even where the implicit one is col_major.

        A `[N, 1]` shape infers blayout=col_major, so `Right` is the one
        destination that still needs an explicit override on top of its implicit
        layout -- this is the case that justifies keeping it.
        """
        span = ir.Span.unknown()
        col_vector = ir.Var("v", ir.TileType([64, 1], DataType.FP32), span)

        assert self._layout_of(tile.move(col_vector, target_memory=ir.MemorySpace.Right).type) == (
            ir.TileLayout.row_major,
            ir.TileLayout.col_major,
            512,
        )

    @pytest.mark.parametrize("space", [d[0] for d in _BOXED_DESTINATIONS])
    def test_move_stamps_the_destination_memory_space(self, space):
        """The deduced type is a view OF the destination, so it must say so."""
        result_type = tile.move(_partial_tile([64, 64], [64, 48]), target_memory=space).type

        assert isinstance(result_type, ir.TileType)
        assert result_type.memory_space == space

    # Vec and Bias share the space-agnostic implicit layout, so they take the
    # source-inherited seed. `destination_dictates_layout` compares layouts
    # rather than naming Vec, so pin both: a future change to either entry in
    # GetImplicitTileLayout must not silently flip this.
    @pytest.mark.parametrize("space", [ir.MemorySpace.Vec, ir.MemorySpace.Bias])
    @pytest.mark.parametrize("valid", [None, [64, 48]], ids=["full-valid", "part-valid"])
    def test_move_to_a_flat_space_inherits_the_source_layout(self, space, valid):
        """A flat destination keeps the source's blayout/slayout, whatever its extent."""
        src = _partial_tile(
            [64, 64],
            valid or [64, 64],
            blayout=ir.TileLayout.col_major,
            slayout=ir.TileLayout.row_major,
        )

        assert self._layout_of(tile.move(src, target_memory=space).type) == (
            ir.TileLayout.col_major,
            ir.TileLayout.row_major,
            512,
        )

    def test_move_explicit_layout_kwargs_still_win(self):
        """blayout/slayout are user overrides and outrank the destination default."""
        ragged = _partial_tile([64, 64], [64, 48])

        call = tile.move(
            ragged,
            target_memory=ir.MemorySpace.Mat,
            blayout=ir.TileLayout.row_major,
            slayout=ir.TileLayout.col_major,
        )

        assert self._layout_of(call.type)[:2] == (ir.TileLayout.row_major, ir.TileLayout.col_major)

    # --- tile.extract -------------------------------------------------------

    @pytest.mark.parametrize(
        "space,blayout,slayout,fractal",
        [
            # L0 destinations use the TEXTRACT-side formats, not tile.move's TMOV ones.
            (ir.MemorySpace.Left, ir.TileLayout.row_major, ir.TileLayout.row_major, 512),
            (ir.MemorySpace.Right, ir.TileLayout.row_major, ir.TileLayout.col_major, 512),
            (ir.MemorySpace.Mat, ir.TileLayout.col_major, ir.TileLayout.row_major, 512),
            (ir.MemorySpace.Vec, ir.TileLayout.row_major, ir.TileLayout.none_box, 512),
            # Acc is reachable via LowerPipelineLoops; the deducer has no Acc
            # branch of its own, so this pins that the Acc NZ view (including
            # fractal 1024) comes from the destination's implicit view.
            (ir.MemorySpace.Acc, ir.TileLayout.col_major, ir.TileLayout.row_major, 1024),
        ],
    )
    def test_extract_layout_is_independent_of_the_source_valid_extent(self, space, blayout, slayout, fractal):
        """Whether the window lands on source padding must not flip the layout."""
        span = ir.Span.unknown()
        full = ir.Var("full", ir.TileType([64, 256], DataType.FP32), span)
        ragged = _partial_tile([64, 256], [64, 100])

        # Same window; over the ragged source it straddles the valid edge (cols
        # 64..96 vs valid 100 -> 32) versus (cols 96..128 -> 4), so the ragged
        # result cannot collapse to an implicit view.
        full_layout = self._layout_of(tile.extract(full, 0, 96, shape=[32, 32], target_memory=space).type)
        ragged_layout = self._layout_of(tile.extract(ragged, 0, 96, shape=[32, 32], target_memory=space).type)

        assert full_layout == ragged_layout == (blayout, slayout, fractal)

    @pytest.mark.parametrize(
        "space",
        [ir.MemorySpace.Left, ir.MemorySpace.Right, ir.MemorySpace.Mat, ir.MemorySpace.Vec],
    )
    def test_extract_stamps_the_destination_memory_space(self, space):
        """Same contract as tile.load: the type names the space it is a view of."""
        span = ir.Span.unknown()
        src = ir.Var("src", ir.TileType([64, 256], DataType.FP32), span)

        result_type = tile.extract(src, 0, 0, shape=[32, 32], target_memory=space).type

        assert isinstance(result_type, ir.TileType)
        assert result_type.memory_space == space

    # --- matmul family ------------------------------------------------------

    def test_matmul_family_carries_the_acc_layout_and_space(self):
        """Every matmul deducer must state the Acc NZ view, not reach it by accident.

        These ops declare `set_output_memory(MemorySpace::Acc)`, and
        `tile.matmul_bias` in particular used to leave the view at the struct
        default (row_major/none_box/512) and land on the Acc layout only because a
        fully-valid view collapses to nullopt and the registry's memory-space
        stamp re-canonicalized it against Acc's implicit view.
        """
        span = ir.Span.unknown()
        lhs = ir.Var("lhs", ir.TileType([16, 32], DataType.FP16), span)
        rhs = ir.Var("rhs", ir.TileType([32, 64], DataType.FP16), span)
        acc = ir.Var("acc", ir.TileType([16, 64], DataType.FP32), span)
        bias = ir.Var("bias", ir.TileType([1, 64], DataType.FP32), span)
        b_lhs = ir.Var("b_lhs", ir.TileType([4, 16, 32], DataType.FP16), span)
        b_rhs = ir.Var("b_rhs", ir.TileType([4, 32, 64], DataType.FP16), span)
        b_acc = ir.Var("b_acc", ir.TileType([4, 16, 64], DataType.FP32), span)

        acc_nz = (ir.TileLayout.col_major, ir.TileLayout.row_major, 1024)
        for name, call in (
            ("matmul", tile.matmul(lhs, rhs)),
            ("matmul_acc", tile.matmul_acc(acc, lhs, rhs)),
            ("matmul_bias", tile.matmul_bias(lhs, rhs, bias)),
            ("batch_matmul", tile.batch_matmul(b_lhs, b_rhs)),
            ("batch_matmul_acc", tile.batch_matmul_acc(b_acc, b_lhs, b_rhs)),
        ):
            result_type = call.type
            assert isinstance(result_type, ir.TileType), name
            assert result_type.memory_space == ir.MemorySpace.Acc, name
            assert self._layout_of(result_type) == acc_nz, name

    def test_matmul_bias_propagates_physical_box_and_logical_valid_shape(self):
        """Biased matmul follows the same padded-box contract as plain matmul."""
        lhs = _partial_tile([32, 64], [16, 64], name="lhs")
        rhs = _partial_tile([64, 32], [64, 16], name="rhs")
        bias = _partial_tile([1, 32], [1, 16], name="bias")

        result_type = tile.matmul_bias(lhs, rhs, bias).type

        assert isinstance(result_type, ir.TileType)
        assert all(isinstance(dim, ir.ConstInt) for dim in result_type.shape)
        assert [cast(ir.ConstInt, dim).value for dim in result_type.shape] == [32, 32]
        assert _valid_of(result_type) == [16, 16]

    def test_gemv_bias_uses_the_shared_product_geometry_contract(self):
        """Shared bias geometry preserves GEMV's padded Acc box and valid N."""
        lhs = _partial_tile([1, 64], [1, 48], name="lhs")
        rhs = _partial_tile([64, 32], [64, 16], name="rhs")
        bias = _partial_tile([1, 32], [1, 24], name="bias")

        result_type = tile.gemv_bias(lhs, rhs, bias).type

        assert isinstance(result_type, ir.TileType)
        assert [cast(ir.ConstInt, dim).value for dim in result_type.shape] == [16, 32]
        assert _valid_of(result_type) == [1, 16]

    def test_matmul_bias_rejects_insufficient_valid_bias_n(self):
        """Bias must cover every valid output column read by the cube."""
        lhs = _partial_tile([32, 64], [16, 64], name="lhs")
        rhs = _partial_tile([64, 32], [64, 24], name="rhs")
        bias = _partial_tile([1, 32], [1, 16], name="bias")

        with pytest.raises(ValueError, match="bias valid N to cover output valid N"):
            tile.matmul_bias(lhs, rhs, bias)

    def test_matmul_bias_rejects_empty_valid_bias_row(self):
        """The cube always reads and broadcasts one logical bias row."""
        lhs = _partial_tile([32, 64], [16, 64], name="lhs")
        rhs = _partial_tile([64, 32], [64, 16], name="rhs")
        bias = _partial_tile([1, 32], [0, 16], name="bias")

        with pytest.raises(ValueError, match="bias valid rows to cover one broadcast row"):
            tile.matmul_bias(lhs, rhs, bias)

    def test_matmul_bias_requires_accumulator_dtype_bias(self):
        """TMATMUL_BIAS requires FP32/INT32 bias to match its Acc output."""
        span = ir.Span.unknown()
        lhs = ir.Var("lhs", ir.TileType([16, 64], DataType.BF16), span)
        rhs = ir.Var("rhs", ir.TileType([64, 32], DataType.BF16), span)
        bias = ir.Var("bias", ir.TileType([1, 32], DataType.BF16), span)

        with pytest.raises(ValueError, match="requires bias dtype fp32"):
            tile.matmul_bias(lhs, rhs, bias)

        int_lhs = ir.Var("int_lhs", ir.TileType([16, 64], DataType.INT8), span)
        int_rhs = ir.Var("int_rhs", ir.TileType([64, 32], DataType.INT8), span)
        int_bias = ir.Var("int_bias", ir.TileType([1, 32], DataType.INT32), span)
        result_type = tile.matmul_bias(int_lhs, int_rhs, int_bias).type
        assert isinstance(result_type, ir.TileType)
        assert result_type.dtype == DataType.INT32


class TestWriteValidRegionUnion:
    """The valid-region union rule shared by tile.assemble and tile.store.

    out_valid[i] = min(shape[i], max(target_valid[i], offset[i] + source_valid[i]))

    accepted only where that candidate provably *is* the union of the target
    rectangle and the written one.
    """

    @staticmethod
    def _partial_tile(shape, valid_shape, name="t", **view_kwargs):
        span = ir.Span.unknown()
        view = ir.TileView(valid_shape=valid_shape, stride=[], start_offset=None, **view_kwargs)
        return ir.Var(name, ir.TileType(shape, DataType.FP32, tile_view=view), span)

    @staticmethod
    def _partial_tensor(shape, valid_shape, name="out"):
        span = ir.Span.unknown()
        view = ir.TensorView(stride=[], layout=ir.TensorLayout.ND, valid_shape=valid_shape)
        return ir.Var(name, ir.TensorType(shape, DataType.FP32, tensor_view=view), span)

    @staticmethod
    def _tile_valid_of(result_type):
        view = result_type.tile_view
        if view is None or not view.valid_shape:
            return [d.value for d in result_type.shape if isinstance(d, ir.ConstInt)]
        return [d.value if isinstance(d, ir.ConstInt) else d for d in view.valid_shape]

    @staticmethod
    def _tensor_valid_of(result_type):
        view = result_type.tensor_view
        if view is None or not view.valid_shape:
            return [d.value for d in result_type.shape if isinstance(d, ir.ConstInt)]
        return [d.value if isinstance(d, ir.ConstInt) else d for d in view.valid_shape]

    # --- tile.assemble ------------------------------------------------------

    def test_assemble_fully_valid_target_stays_implicit(self):
        """A full target keeps the implicit view its layout depends on."""
        span = ir.Span.unknown()
        target = ir.Var("dst", ir.TileType([64, 128], DataType.FP32), span)
        source = self._partial_tile([16, 128], [12, 128], name="src")

        result_type = tile.assemble(target, source, [8, 0]).type

        assert isinstance(result_type, ir.TileType)
        assert result_type.tile_view is None

    def test_assemble_padded_source_does_not_disturb_the_target_beyond_it(self):
        """An ISA-padded source moves its valid region, not its allocation.

        Codegen emits ``pto.subview %dst[...] sizes [R, C] valid [Vr, Vc]`` and
        moves into that view, so the band between the source's valid extent and
        its allocation is addressed but not transferred. The target keeps what it
        had there, which is why a fully valid target stays fully valid and the
        ISA-padding idiom of PR #1528 is accepted.
        """
        span = ir.Span.unknown()
        target = ir.Var("dst", ir.TileType([64, 128], DataType.FP32), span)
        # 8 of 16 rows real -- the classic fixed-width staging tile.
        source = self._partial_tile([16, 128], [8, 128], name="src")

        result_type = tile.assemble(target, source, [0, 0]).type

        assert isinstance(result_type, ir.TileType)
        assert result_type.tile_view is None

    def test_assemble_padded_source_appends_only_its_valid_region(self):
        """The append grows by the source's real extent, not its allocation."""
        target = self._partial_tile([64, 128], [20, 128], name="dst")
        source = self._partial_tile([16, 128], [8, 128], name="src")

        result_type = tile.assemble(target, source, [20, 0]).type

        # 8 real rows land at 20-27; the allocation's remaining 8 rows move nothing.
        assert self._tile_valid_of(result_type) == [28, 128]

    def test_assemble_contiguous_growth_extends_one_dimension(self):
        """An append that abuts the target grows exactly that axis."""
        target = self._partial_tile([64, 128], [20, 128], name="dst")
        source = self._partial_tile([12, 128], [12, 128], name="src")

        result_type = tile.assemble(target, source, [20, 0]).type

        assert self._tile_valid_of(result_type) == [32, 128]

    def test_assemble_empty_source_is_a_no_op(self):
        """Writing an empty region leaves the target exactly as it was."""
        target = self._partial_tile([64, 128], [20, 128], name="dst")
        source = self._partial_tile([16, 128], [0, 128], name="src")

        result_type = tile.assemble(target, source, [20, 0]).type

        assert self._tile_valid_of(result_type) == [20, 128]

    def test_assemble_empty_target_is_initialized_from_the_origin(self):
        """An empty accumulator takes the written region as its valid region."""
        target = self._partial_tile([64, 128], [0, 128], name="dst")
        source = self._partial_tile([16, 128], [12, 128], name="src")

        result_type = tile.assemble(target, source, [0, 0]).type

        assert self._tile_valid_of(result_type) == [12, 128]

    def test_assemble_gap_rejects(self):
        """A write starting past the target's edge leaves an unrepresentable hole."""
        target = self._partial_tile([64, 128], [20, 128], name="dst")
        source = self._partial_tile([12, 128], [12, 128], name="src")

        with pytest.raises(ValueError, match="leaves a gap in dimension 0"):
            tile.assemble(target, source, [24, 0])

    def test_assemble_l_shape_rejects(self):
        """Growing two axes at once is not one origin-anchored rectangle."""
        target = self._partial_tile([64, 256], [20, 64], name="dst")
        source = self._partial_tile([32, 128], [12, 80], name="src")

        with pytest.raises(ValueError, match="dimensions 0 and 1 at once"):
            tile.assemble(target, source, [20, 0])

    def test_assemble_validates_the_physical_source_subview(self):
        """``pto.tinsert`` copies the whole subview, so all of it must fit.

        The tensor.assemble counterpart accepts this same write, because a tensor
        transfer moves only the source's valid region.
        """
        span = ir.Span.unknown()
        target = ir.Var("dst", ir.TileType([64, 128], DataType.FP32), span)
        # 48 rows allocated, only 8 real; the allocation overhangs the target.
        source = self._partial_tile([48, 128], [8, 128], name="src")

        with pytest.raises(ValueError, match="writes past the end of dimension 0"):
            tile.assemble(target, source, [56, 0])

    def test_assemble_bounds_rejection_names_the_subview_rule(self):
        """tile.assemble's overhang error explains why the allocation must fit."""
        span = ir.Span.unknown()
        target = ir.Var("dst", ir.TileType([64, 128], DataType.FP32), span)
        source = self._partial_tile([48, 128], [8, 128], name="src")

        with pytest.raises(ValueError, match="copies the whole source subview"):
            tile.assemble(target, source, [56, 0])

    def test_assemble_keeps_the_target_layout_while_narrowing(self):
        """A partial union must not cost the target its layout metadata."""
        target = self._partial_tile([64, 128], [20, 128], name="dst", blayout=ir.TileLayout.col_major)
        source = self._partial_tile([12, 128], [12, 128], name="src")

        result_type = tile.assemble(target, source, [20, 0]).type

        assert isinstance(result_type, ir.TileType)
        view = result_type.tile_view
        assert view is not None
        assert view.blayout == ir.TileLayout.col_major
        assert self._tile_valid_of(result_type) == [32, 128]

    def test_assemble_negative_offset_rejects(self):
        """A write must start inside its target."""
        span = ir.Span.unknown()
        target = self._partial_tile([64, 128], [20, 128], name="dst")
        source = self._partial_tile([12, 128], [12, 128], name="src")
        neg = ir.ConstInt(-4, DataType.INDEX, span)

        with pytest.raises(ValueError, match="provably negative"):
            tile.assemble(target, source, [neg, 0])

    # --- tile.store ---------------------------------------------------------

    def test_store_into_fully_valid_destination_is_unchanged(self):
        """The overwhelmingly common store keeps the destination type it had."""
        span = ir.Span.unknown()
        out = ir.Var("out", ir.TensorType([64, 128], DataType.FP32), span)
        src = self._partial_tile([16, 128], [12, 128], name="src")

        result_type = tile.store(src, [8, 0], out).type

        assert isinstance(result_type, ir.TensorType)
        assert result_type.tensor_view is None

    def test_store_unions_into_a_partially_valid_destination(self):
        """A store appends to the destination's valid region."""
        out = self._partial_tensor([64, 128], [20, 128])
        src = self._partial_tile([16, 128], [12, 128], name="src")

        result_type = tile.store(src, [20, 0], out).type

        assert self._tensor_valid_of(result_type) == [32, 128]

    def test_store_transfers_only_the_tile_valid_region(self):
        """The DMA moves the real extent, so a padded tile is bounded by it."""
        span = ir.Span.unknown()
        out = ir.Var("out", ir.TensorType([64, 128], DataType.FP32), span)
        # 64 rows allocated, 8 real, landing on the destination's last 8 rows.
        src = self._partial_tile([64, 128], [8, 128], name="src")

        result_type = tile.store(src, [56, 0], out).type

        assert isinstance(result_type, ir.TensorType)
        assert result_type.tensor_view is None

    def test_store_validates_destination_bounds(self):
        """A transfer that runs off the destination rejects."""
        span = ir.Span.unknown()
        out = ir.Var("out", ir.TensorType([64, 128], DataType.FP32), span)
        src = self._partial_tile([32, 128], [32, 128], name="src")

        with pytest.raises(ValueError, match="writes past the end of dimension 0"):
            tile.store(src, [56, 0], out)

    def test_store_bounds_rejection_names_the_layout_remedy(self):
        """An overhang is usually a coordinate mismatch, so the error says so.

        A tile read through a DN view is transposed; storing it into the
        destination's untransposed shape overflows one axis while under-filling
        the other. ``tile.store`` converts no layouts (RFC #1300 P7), so the
        diagnostic points at taking the matching view of the destination.
        """
        span = ir.Span.unknown()
        out = ir.Var("out", ir.TensorType([8, 16], DataType.FP32), span)
        src = self._partial_tile([16, 8], [16, 8], name="src")

        with pytest.raises(ValueError, match="performs no layout conversion"):
            tile.store(src, [0, 0], out)

    def test_store_gap_rejects(self):
        """A store that does not abut the destination's valid region rejects."""
        out = self._partial_tensor([64, 128], [20, 128])
        src = self._partial_tile([16, 128], [12, 128], name="src")

        with pytest.raises(ValueError, match="leaves a gap in dimension 0"):
            tile.store(src, [24, 0], out)

    def test_store_leaves_the_nd_partition_form_alone(self):
        """The ND ``shapes`` operand is a collapsed-dims descriptor, not a rectangle.

        FlattenTileNdTo2D builds it as leading 1s followed by the pre-flatten tile
        shape, so its leading extent can be a product of several destination axes
        and need not fit the matching one. Reading it as an origin-anchored
        rectangle would mis-bound the write and place the union on the wrong axes,
        so the destination type is passed through untouched.
        """
        out = self._partial_tensor([2, 3, 16, 64], [1, 3, 16, 64])
        src = self._partial_tile([16, 64], [16, 64], name="src")

        result_type = tile.store(src, [1, 0, 0, 0], out, [1, 3, 16, 64]).type

        assert self._tensor_valid_of(result_type) == [1, 3, 16, 64]

    def test_store_accepts_a_collapsed_nd_partition(self):
        """A partition extent may exceed the destination axis it nominally sits on.

        This is the shape ``tests/st/runtime/ops/test_gather.py`` produces: a
        ``[2, 3, 8]`` gather whose lowering collapses the leading dims to a
        ``[6, 8]`` tile, stored as partition ``[1, 6, 8]`` where 6 > 3.
        """
        span = ir.Span.unknown()
        out = ir.Var("out", ir.TensorType([2, 3, 8], DataType.FP32), span)
        src = self._partial_tile([6, 8], [6, 8], name="src")

        result_type = tile.store(src, [0, 0, 0], out, [1, 6, 8]).type

        assert isinstance(result_type, ir.TensorType)
        assert result_type.tensor_view is None

    def test_store_rank_mismatch_keeps_its_previous_result(self):
        """A reinterpreting store is not a rectangle on the destination axes."""
        span = ir.Span.unknown()
        out = ir.Var("out", ir.TensorType([1, 16, 64], DataType.FP32), span)
        src = self._partial_tile([16, 64], [8, 64], name="src")

        result_type = tile.store(src, [0, 0], out).type

        assert isinstance(result_type, ir.TensorType)
        assert result_type.tensor_view is None


class TestTileSort32Ops:
    """Type-inference coverage for TSORT32's packed value-index output."""

    @pytest.mark.parametrize(
        ("dtype", "expected_width"),
        [(DataType.FP32, 64), (DataType.FP16, 128)],
    )
    def test_output_width_depends_on_dtype(self, dtype, expected_width):
        span = ir.Span.unknown()
        src = ir.Var("src", ir.TileType([1, 32], dtype), span)
        idx = ir.Var("idx", ir.TileType([1, 32], DataType.UINT32), span)

        result_type = tile.sort32(src, idx).type

        assert isinstance(result_type, ir.TileType)
        assert result_type.dtype == dtype
        assert result_type.shape == [1, expected_width]
        assert _valid_of(result_type) == [1, expected_width]

    @pytest.mark.parametrize(
        ("dtype", "factor", "physical_width"),
        [(DataType.FP32, 2, 128), (DataType.FP16, 4, 256)],
    )
    def test_scales_symbolic_valid_width(self, dtype, factor, physical_width):
        span = ir.Span.unknown()
        valid_cols = ir.Var("valid_cols", ir.ScalarType(DataType.INDEX), span)
        src_view = ir.TileView(valid_shape=[1, valid_cols])
        idx_view = ir.TileView(valid_shape=[1, valid_cols])
        src = ir.Var("src", ir.TileType([1, 64], dtype, tile_view=src_view), span)
        idx = ir.Var("idx", ir.TileType([1, 64], DataType.UINT32, tile_view=idx_view), span)

        result_type = tile.sort32(src, idx).type

        assert isinstance(result_type, ir.TileType)
        assert result_type.shape == [1, physical_width]
        valid_width = result_type.get_effective_tile_view().valid_shape[1]
        assert isinstance(valid_width, ir.Mul)
        assert valid_width.left is valid_cols
        assert isinstance(valid_width.right, ir.ConstInt)
        assert valid_width.right.value == factor


class TestB03TriAndGatherOps:
    """IR contracts for TTRI, TGATHERB, and MGATHER."""

    @staticmethod
    def _tile(name, shape, dtype, valid_shape=None):
        span = ir.Span.unknown()
        view = None if valid_shape is None else ir.TileView(valid_shape=valid_shape)
        return ir.Var(name, ir.TileType(shape, dtype, tile_view=view), span)

    @staticmethod
    def _assert_program_round_trip(program):
        printed = str(program)
        reparsed = pl.parse_program(printed)
        ir.assert_structural_equal(program, reparsed)
        return printed

    def test_tri_preserves_physical_and_partial_valid_shape(self):
        call = tile.tri(1, [16, 32], valid_shape=[9, 21], dtype=DataType.FP16, upper=True)

        assert call.op.name == ir.get_op("tile.tri").name
        assert dict(call.kwargs) == {"dtype": DataType.FP16, "upper": True}
        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        assert [dim.value for dim in result_type.shape if isinstance(dim, ir.ConstInt)] == [16, 32]
        assert _valid_of(result_type) == [9, 21]
        assert result_type.dtype == DataType.FP16

    @pytest.mark.parametrize(
        "dtype",
        [
            DataType.INT16,
            DataType.INT32,
            DataType.UINT16,
            DataType.UINT32,
            DataType.FP16,
            DataType.FP32,
        ],
    )
    def test_tri_supported_dtypes(self, dtype):
        assert _tile_result_dtype(tile.tri(0, [8, 16], dtype=dtype)) == dtype

    def test_tri_rejects_invalid_valid_shape(self):
        with pytest.raises(ValueError, match="valid_shape"):
            tile.tri(0, [8, 16], valid_shape=[9, 16])

    def test_tri_print_parse_round_trip(self):
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                out: pl.Tensor[[16, 32], pl.FP16],
            ) -> pl.Tensor[[16, 32], pl.FP16]:
                result = pl.tile.tri(
                    1,
                    [16, 32],
                    valid_shape=[9, 21],
                    dtype=pl.FP16,
                    upper=True,
                )
                return pl.store(result, [0, 0], out)

        printed = self._assert_program_round_trip(Prog)
        assert "valid_shape=[9, 21]" in printed
        assert "upper=True" in printed

    def test_tri_rejects_invalid_scalar_shape_and_dtype_contracts(self):
        span = ir.Span.unknown()
        fp_diagonal = ir.Var("diagonal", ir.ScalarType(DataType.FP32), span)
        dynamic_dim = ir.Var("dynamic_dim", ir.ScalarType(DataType.INDEX), span)

        with pytest.raises(ValueError, match="INT32 scalar"):
            tile.tri(fp_diagonal, [8, 16])
        with pytest.raises(ValueError, match="requires a 2D shape"):
            tile.tri(0, [16])
        with pytest.raises(ValueError, match="compile-time constant"):
            tile.tri(0, [dynamic_dim, 16])
        with pytest.raises(ValueError, match="must be positive"):
            tile.tri(0, [0, 16])
        with pytest.raises(ValueError, match="requires dtype"):
            tile.tri(0, [8, 16], dtype=DataType.BOOL)
        with pytest.raises(ValueError, match="valid_shape rank"):
            tile.tri(0, [8, 16], valid_shape=[8])
        with pytest.raises(ValueError, match=r"0 < valid_shape\[0\]"):
            tile.tri(0, [8, 16], valid_shape=[0, 16])

    def test_gatherb_expands_block_offsets_to_output_elements(self):
        src = self._tile("src", [16, 64], DataType.FP16, [16, 64])
        offset = self._tile("offset", [8, 16], DataType.UINT32, [5, 9])

        call = tile.gatherb(src, offset)

        assert call.op.name == ir.get_op("tile.gatherb").name
        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        assert [dim.value for dim in result_type.shape if isinstance(dim, ir.ConstInt)] == [8, 256]
        assert _valid_of(result_type) == [5, 144]
        assert result_type.dtype == DataType.FP16

    def test_gatherb_supports_distinct_output_dtype(self):
        src = self._tile("src", [16, 64], DataType.FP16, [16, 64])
        offset = self._tile("offset", [8, 16], DataType.UINT32, [5, 9])

        result_type = tile.gatherb(src, offset, output_dtype=DataType.FP32).type

        assert isinstance(result_type, ir.TileType)
        assert [dim.value for dim in result_type.shape if isinstance(dim, ir.ConstInt)] == [8, 128]
        assert _valid_of(result_type) == [5, 72]
        assert result_type.dtype == DataType.FP32

    def test_gatherb_scales_symbolic_valid_columns(self):
        span = ir.Span.unknown()
        valid_cols = ir.Var("valid_cols", ir.ScalarType(DataType.INDEX), span)
        src = self._tile("src", [16, 64], DataType.FP16, [16, 64])
        offset_type = ir.TileType(
            [8, 16],
            DataType.UINT32,
            tile_view=ir.TileView(valid_shape=[5, valid_cols]),
        )
        offset = ir.Var("offset", offset_type, span)

        result_type = tile.gatherb(src, offset).type

        assert isinstance(result_type, ir.TileType)
        valid_shape = result_type.get_effective_tile_view().valid_shape
        assert isinstance(valid_shape[1], ir.Mul)
        assert valid_shape[1].left is valid_cols
        assert isinstance(valid_shape[1].right, ir.ConstInt)
        assert valid_shape[1].right.value == 16

    def test_gatherb_rejects_non_uint32_offsets(self):
        src = self._tile("src", [8, 16], DataType.FP16)
        offset = self._tile("offset", [8, 16], DataType.INT32)
        with pytest.raises(ValueError, match="UINT32"):
            tile.gatherb(src, offset)

    def test_gatherb_rejects_unaligned_offset_rows(self):
        src = self._tile("src", [8, 16], DataType.FP16)
        offset = self._tile("offset", [8, 7], DataType.UINT32)
        with pytest.raises(ValueError, match="multiple of 8"):
            tile.gatherb(src, offset)

    def test_gatherb_print_parse_round_trip(self):
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                src: pl.Tensor[[16, 64], pl.FP16],
                offsets: pl.Tensor[[8, 8], pl.UINT32],
                out: pl.Tensor[[8, 64], pl.FP32],
            ) -> pl.Tensor[[8, 64], pl.FP32]:
                src_tile = pl.load(src, [0, 0], [16, 64])
                offset_tile = pl.load(offsets, [0, 0], [8, 8], valid_shape=[5, 5])
                gathered = pl.tile.gatherb(src_tile, offset_tile, output_dtype=pl.FP32)
                return pl.store(gathered, [0, 0], out)

        printed = self._assert_program_round_trip(Prog)
        assert "output_dtype=pl.FP32" in printed

    def test_gatherb_rejects_invalid_dtype_rank_and_static_shape_contracts(self):
        span = ir.Span.unknown()
        dynamic_cols = ir.Var("dynamic_cols", ir.ScalarType(DataType.INDEX), span)
        valid_src = self._tile("src", [8, 16], DataType.FP16)
        valid_offset = self._tile("offset", [8, 8], DataType.UINT32)

        with pytest.raises(ValueError, match="src dtype"):
            tile.gatherb(self._tile("src", [8, 16], DataType.BOOL), valid_offset)
        with pytest.raises(ValueError, match="output_dtype"):
            tile.gatherb(valid_src, valid_offset, output_dtype=DataType.BOOL)
        with pytest.raises(ValueError, match="2D src"):
            tile.gatherb(self._tile("src", [128], DataType.FP16), valid_offset)
        with pytest.raises(ValueError, match="2D offset"):
            tile.gatherb(valid_src, self._tile("offset", [64], DataType.UINT32))
        dynamic_offset = ir.Var(
            "dynamic_offset",
            ir.TileType([ir.ConstInt(8, DataType.INDEX, span), dynamic_cols], DataType.UINT32),
            span,
        )
        with pytest.raises(ValueError, match="static offset columns"):
            tile.gatherb(valid_src, dynamic_offset)
        with pytest.raises(ValueError, match="positive multiple of 8"):
            tile.gatherb(valid_src, self._tile("offset", [8, 0], DataType.UINT32))

    def test_mgather_row_mode_shapes_from_index_and_table(self):
        span = ir.Span.unknown()
        mem = ir.Var("mem", ir.TensorType([64, 32], DataType.BF16), span)
        idx = self._tile("idx", [1, 16], DataType.INT32, [1, 9])

        call = tile.mgather(mem, idx)

        assert call.op.name == ir.get_op("tile.mgather").name
        assert dict(call.kwargs) == {"coalesce": 0}
        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        assert [dim.value for dim in result_type.shape if isinstance(dim, ir.ConstInt)] == [16, 32]
        assert _valid_of(result_type) == [9, 32]
        assert result_type.dtype == DataType.BF16

    def test_mgather_elem_mode_preserves_index_region(self):
        span = ir.Span.unknown()
        mem = ir.Var("mem", ir.TensorType([256], DataType.INT16), span)
        idx = self._tile("idx", [8, 32], DataType.INT32, [5, 19])

        call = tile.mgather(mem, idx, coalesce="elem")

        assert dict(call.kwargs) == {"coalesce": 1}
        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        assert [dim.value for dim in result_type.shape if isinstance(dim, ir.ConstInt)] == [8, 32]
        assert _valid_of(result_type) == [5, 19]

    def test_mgather_mat_row_uses_gm_index_and_nz_result(self):
        span = ir.Span.unknown()
        mem = ir.Var("mem", ir.TensorType([64, 32], DataType.FP16), span)
        idx = ir.Var("idx", ir.TensorType([1, 16], DataType.INT32), span)

        call = tile.mgather(mem, idx, target_memory=ir.MemorySpace.Mat)

        assert dict(call.kwargs) == {
            "coalesce": 0,
            "target_memory": ir.MemorySpace.Mat,
        }
        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        assert [dim.value for dim in result_type.shape if isinstance(dim, ir.ConstInt)] == [16, 32]
        view = result_type.get_effective_tile_view()
        assert view.blayout == ir.TileLayout.col_major
        assert view.slayout == ir.TileLayout.row_major

    @pytest.mark.parametrize("coalesce", ["row", "elem"])
    def test_mgather_mat_preserves_explicit_valid_shape(self, coalesce):
        span = ir.Span.unknown()
        mem_shape = [64, 32] if coalesce == "row" else [512]
        mem = ir.Var("mem", ir.TensorType(mem_shape, DataType.FP16), span)
        idx_shape = [1, 16] if coalesce == "row" else [16, 32]
        idx = ir.Var("idx", ir.TensorType(idx_shape, DataType.INT32), span)
        scratch = ir.Var("scratch", ir.TensorType([512], DataType.FP16), span) if coalesce == "elem" else None

        result_type = tile.mgather(
            mem,
            idx,
            coalesce=coalesce,
            target_memory=ir.MemorySpace.Mat,
            scratch=scratch,
            valid_shape=[9, 21],
        ).type

        assert isinstance(result_type, ir.TileType)
        assert [dim.value for dim in result_type.shape if isinstance(dim, ir.ConstInt)] == [16, 32]
        assert _valid_of(result_type) == [9, 21]

    def test_mgather_mat_row_valid_shape_round_trips(self):
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                mem: pl.Tensor[[64, 32], pl.FP16],
                idx: pl.Tensor[[1, 16], pl.INT32],
                eye: pl.Tensor[[16, 16], pl.FP16],
                out: pl.Tensor[[16, 32], pl.FP32],
            ) -> pl.Tensor[[16, 32], pl.FP32]:
                gathered = pl.tile.mgather(
                    mem,
                    idx,
                    target_memory=pl.MemorySpace.Mat,
                    valid_shape=[9, 21],
                )
                eye_tile = pl.load(
                    eye, [0, 0], [16, 16], valid_shape=[16, 9], target_memory=pl.MemorySpace.Mat
                )
                product = pl.matmul(eye_tile, gathered)
                return pl.store(product, [0, 0], out)

        printed = self._assert_program_round_trip(Prog)
        assert "valid_shape=[9, 21]" in printed

    def test_mgather_mat_elem_scratch_round_trips(self):
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                mem: pl.Tensor[[512], pl.FP16],
                idx: pl.Tensor[[16, 32], pl.INT32],
                scratch: pl.Tensor[[512], pl.FP16],
                eye: pl.Tensor[[16, 16], pl.FP16],
                out: pl.Tensor[[16, 32], pl.FP32],
            ) -> pl.Tensor[[16, 32], pl.FP32]:
                gathered = pl.tile.mgather(
                    mem,
                    idx,
                    coalesce="elem",
                    target_memory=pl.MemorySpace.Mat,
                    scratch=scratch,
                )
                eye_tile = pl.load(
                    eye, [0, 0], [16, 16], valid_shape=[16, 16], target_memory=pl.MemorySpace.Mat
                )
                product = pl.matmul(eye_tile, gathered)
                return pl.store(product, [0, 0], out)

        printed = self._assert_program_round_trip(Prog)
        assert "scratch=scratch" in printed

    def test_mgather_mat_elem_scratch_and_valid_shape_round_trip(self):
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                mem: pl.Tensor[[512], pl.FP16],
                idx: pl.Tensor[[16, 32], pl.INT32],
                scratch: pl.Tensor[[512], pl.FP16],
                eye: pl.Tensor[[16, 16], pl.FP16],
                out: pl.Tensor[[16, 32], pl.FP32],
            ) -> pl.Tensor[[16, 32], pl.FP32]:
                gathered = pl.tile.mgather(
                    mem,
                    idx,
                    coalesce="elem",
                    target_memory=pl.MemorySpace.Mat,
                    scratch=scratch,
                    valid_shape=[9, 21],
                )
                eye_tile = pl.load(
                    eye, [0, 0], [16, 16], valid_shape=[16, 9], target_memory=pl.MemorySpace.Mat
                )
                product = pl.matmul(eye_tile, gathered)
                return pl.store(product, [0, 0], out)

        printed = self._assert_program_round_trip(Prog)
        assert "scratch=scratch" in printed
        assert "valid_shape=[9, 21]" in printed

    def test_mgather_mat_rejects_invalid_valid_shape(self):
        span = ir.Span.unknown()
        mem = ir.Var("mem", ir.TensorType([64, 32], DataType.FP16), span)
        idx = ir.Var("idx", ir.TensorType([1, 16], DataType.INT32), span)

        with pytest.raises(ValueError, match="valid_shape must have rank 2"):
            tile.mgather(
                mem,
                idx,
                target_memory=ir.MemorySpace.Mat,
                valid_shape=[9],
            )
        with pytest.raises(ValueError, match=r"0 < Mat valid_shape\[1\]"):
            tile.mgather(
                mem,
                idx,
                target_memory=ir.MemorySpace.Mat,
                valid_shape=[9, 33],
            )

    def test_mgather_mat_rejects_non_nd_source(self):
        span = ir.Span.unknown()
        dn_view = ir.TensorView(stride=[], layout=ir.TensorLayout.DN)
        mem = ir.Var(
            "mem",
            ir.TensorType([64, 32], DataType.FP16, tensor_view=dn_view),
            span,
        )
        idx = ir.Var("idx", ir.TensorType([1, 16], DataType.INT32), span)

        with pytest.raises(ValueError, match="requires mem to use ND tensor layout"):
            tile.mgather(mem, idx, target_memory=ir.MemorySpace.Mat)

        mem = ir.Var("mem", ir.TensorType([64, 32], DataType.FP16), span)
        idx = ir.Var(
            "idx",
            ir.TensorType([1, 16], DataType.INT32, tensor_view=dn_view),
            span,
        )
        with pytest.raises(ValueError, match="requires idx to use ND tensor layout"):
            tile.mgather(mem, idx, target_memory=ir.MemorySpace.Mat)

    def test_mgather_mat_elem_requires_matching_gm_scratch(self):
        span = ir.Span.unknown()
        mem = ir.Var("mem", ir.TensorType([256], DataType.FP16), span)
        idx = ir.Var("idx", ir.TensorType([16, 16], DataType.INT32), span)
        scratch = ir.Var("scratch", ir.TensorType([256], DataType.FP16), span)

        result_type = tile.mgather(
            mem,
            idx,
            coalesce="elem",
            target_memory=ir.MemorySpace.Mat,
            scratch=scratch,
        ).type

        assert isinstance(result_type, ir.TileType)
        assert [dim.value for dim in result_type.shape if isinstance(dim, ir.ConstInt)] == [16, 16]
        with pytest.raises(ValueError, match="requires GM scratch"):
            tile.mgather(mem, idx, coalesce="elem", target_memory=ir.MemorySpace.Mat)

        wrong_scratch = ir.Var("wrong", ir.TensorType([256], DataType.INT32), span)
        with pytest.raises(ValueError, match="dtype must match"):
            tile.mgather(
                mem,
                idx,
                coalesce="elem",
                target_memory=ir.MemorySpace.Mat,
                scratch=wrong_scratch,
            )

        short_scratch = ir.Var("short", ir.TensorType([255], DataType.FP16), span)
        with pytest.raises(ValueError, match="at least 256 elements"):
            tile.mgather(
                mem,
                idx,
                coalesce="elem",
                target_memory=ir.MemorySpace.Mat,
                scratch=short_scratch,
            )

        noncontiguous_view = ir.TensorView(stride=[32, 1], layout=ir.TensorLayout.ND)
        noncontiguous_scratch = ir.Var(
            "noncontiguous",
            ir.TensorType([16, 16], DataType.FP16, tensor_view=noncontiguous_view),
            span,
        )
        with pytest.raises(ValueError, match="contiguous ND"):
            tile.mgather(
                mem,
                idx,
                coalesce="elem",
                target_memory=ir.MemorySpace.Mat,
                scratch=noncontiguous_scratch,
            )

        singleton_view = ir.TensorView(stride=[512, 1], layout=ir.TensorLayout.ND)
        singleton_scratch = ir.Var(
            "singleton",
            ir.TensorType([1, 256], DataType.FP16, tensor_view=singleton_view),
            span,
        )
        singleton_result = tile.mgather(
            mem,
            idx,
            coalesce="elem",
            target_memory=ir.MemorySpace.Mat,
            scratch=singleton_scratch,
        )
        assert isinstance(singleton_result.type, ir.TileType)

    def test_mgather_mat_elem_rejects_direct_scratch_aliases(self):
        span = ir.Span.unknown()
        fp_mem = ir.Var("fp_mem", ir.TensorType([256], DataType.FP16), span)
        idx = ir.Var("idx", ir.TensorType([16, 16], DataType.INT32), span)

        with pytest.raises(ValueError, match="must not alias mem or idx"):
            tile.mgather(
                fp_mem,
                idx,
                coalesce="elem",
                target_memory=ir.MemorySpace.Mat,
                scratch=fp_mem,
            )

        int_mem = ir.Var("int_mem", ir.TensorType([256], DataType.INT32), span)
        with pytest.raises(ValueError, match="must not alias mem or idx"):
            tile.mgather(
                int_mem,
                idx,
                coalesce="elem",
                target_memory=ir.MemorySpace.Mat,
                scratch=idx,
            )

    def test_mgather_rejects_scratch_outside_mat_elem(self):
        span = ir.Span.unknown()
        mem = ir.Var("mem", ir.TensorType([64, 32], DataType.FP16), span)
        vec_idx = self._tile("vec_idx", [1, 16], DataType.INT32)
        mat_idx = ir.Var("mat_idx", ir.TensorType([1, 16], DataType.INT32), span)
        scratch = ir.Var("scratch", ir.TensorType([512], DataType.FP16), span)

        with pytest.raises(ValueError, match="permits scratch only for Mat elem mode"):
            tile.mgather(mem, vec_idx, scratch=scratch)
        with pytest.raises(ValueError, match="Mat row mode accepts only an optional valid_shape"):
            tile.mgather(
                mem,
                mat_idx,
                target_memory=ir.MemorySpace.Mat,
                scratch=scratch,
                valid_shape=[16, 32],
            )

    def test_mgather_mat_rejects_non_nz_aligned_result(self):
        span = ir.Span.unknown()
        mem = ir.Var("mem", ir.TensorType([64, 17], DataType.FP16), span)
        idx = ir.Var("idx", ir.TensorType([1, 15], DataType.INT32), span)

        with pytest.raises(ValueError, match="rows must be a multiple of 16"):
            tile.mgather(mem, idx, target_memory=ir.MemorySpace.Mat)

        idx = ir.Var("idx", ir.TensorType([1, 16], DataType.INT32), span)
        with pytest.raises(ValueError, match="cols must be a multiple of 16"):
            tile.mgather(mem, idx, target_memory=ir.MemorySpace.Mat)

    def test_mgather_memory_space_selects_index_contract(self):
        span = ir.Span.unknown()
        mem = ir.Var("mem", ir.TensorType([64, 32], DataType.FP32), span)
        idx_tile = self._tile("idx_tile", [1, 16], DataType.INT32)
        idx_tensor = ir.Var("idx_tensor", ir.TensorType([1, 16], DataType.INT32), span)

        with pytest.raises(ValueError, match="GM TensorType"):
            tile.mgather(mem, idx_tile, target_memory=ir.MemorySpace.Mat)
        with pytest.raises(ValueError, match="TileType"):
            tile.mgather(mem, idx_tensor)

    def test_mgather_rejects_invalid_index_ranks_and_row_orientations(self):
        span = ir.Span.unknown()
        mem = ir.Var("mem", ir.TensorType([64, 32], DataType.FP32), span)

        with pytest.raises(ValueError, match="requires a 2D idx tile"):
            tile.mgather(mem, self._tile("vec_rank3", [1, 1, 16], DataType.INT32))
        with pytest.raises(ValueError, match=r"requires a \[1, R\] or \[R, 1\] idx shape"):
            tile.mgather(mem, self._tile("vec_matrix", [2, 8], DataType.INT32))

        mat_rank3 = ir.Var("mat_rank3", ir.TensorType([1, 1, 16], DataType.INT32), span)
        with pytest.raises(ValueError, match="requires a 2D Mat idx tensor"):
            tile.mgather(mem, mat_rank3, target_memory=ir.MemorySpace.Mat)
        mat_column = ir.Var("mat_column", ir.TensorType([16, 1], DataType.INT32), span)
        with pytest.raises(ValueError, match=r"requires a \[1, R\] GM idx tensor"):
            tile.mgather(mem, mat_column, target_memory=ir.MemorySpace.Mat)

    def test_mgather_mat_elem_rejects_non_gm_or_dynamic_scratch(self):
        span = ir.Span.unknown()
        dynamic_dim = ir.Var("dynamic_dim", ir.ScalarType(DataType.INDEX), span)
        mem = ir.Var("mem", ir.TensorType([512], DataType.FP16), span)
        idx = ir.Var("idx", ir.TensorType([16, 32], DataType.INT32), span)

        tile_scratch = self._tile("tile_scratch", [16, 32], DataType.FP16)
        with pytest.raises(ValueError, match="scratch must be a GM tensor"):
            tile.mgather(
                mem,
                idx,
                coalesce="elem",
                target_memory=ir.MemorySpace.Mat,
                scratch=tile_scratch,
            )

        dynamic_scratch = ir.Var("dynamic_scratch", ir.TensorType([dynamic_dim], DataType.FP16), span)
        with pytest.raises(ValueError, match="scratch shape must be static"):
            tile.mgather(
                mem,
                idx,
                coalesce="elem",
                target_memory=ir.MemorySpace.Mat,
                scratch=dynamic_scratch,
            )

    @pytest.mark.parametrize("target_memory", [ir.MemorySpace.Vec, ir.MemorySpace.Mat])
    def test_mgather_rejects_uint32_index_for_pinned_ptoas(self, target_memory):
        span = ir.Span.unknown()
        mem = ir.Var("mem", ir.TensorType([64, 32], DataType.FP32), span)
        idx = (
            self._tile("idx", [1, 16], DataType.UINT32)
            if target_memory == ir.MemorySpace.Vec
            else ir.Var("idx", ir.TensorType([1, 16], DataType.UINT32), span)
        )

        with pytest.raises(ValueError, match="INT32"):
            tile.mgather(mem, idx, target_memory=target_memory)

    def test_mgather_row_rejects_rank_one_mem(self):
        span = ir.Span.unknown()
        mem = ir.Var("mem", ir.TensorType([256], DataType.INT16), span)
        idx = self._tile("idx", [1, 8], DataType.INT32)

        with pytest.raises(ValueError, match="mem rank >= 2"):
            tile.mgather(mem, idx, coalesce="row")

    @pytest.mark.parametrize(
        "dtype",
        [
            DataType.INT8,
            DataType.UINT8,
            DataType.INT16,
            DataType.UINT16,
            DataType.INT32,
            DataType.UINT32,
            DataType.FP16,
            DataType.BF16,
            DataType.FP32,
            DataType.FP8E4M3FN,
            DataType.FP8E5M2,
            DataType.HF8,
        ],
    )
    @pytest.mark.parametrize("target_memory", [ir.MemorySpace.Vec, ir.MemorySpace.Mat])
    def test_mgather_supported_payload_dtypes(self, dtype, target_memory):
        span = ir.Span.unknown()
        mem = ir.Var("mem", ir.TensorType([64, 32], dtype), span)
        idx = (
            self._tile("idx", [1, 16], DataType.INT32)
            if target_memory == ir.MemorySpace.Vec
            else ir.Var("idx", ir.TensorType([1, 16], DataType.INT32), span)
        )

        assert _tile_result_dtype(tile.mgather(mem, idx, target_memory=target_memory)) == dtype

    def test_mgather_row_accepts_a5_column_vector_index(self):
        span = ir.Span.unknown()
        mem = ir.Var("mem", ir.TensorType([64, 32], DataType.FP32), span)
        idx = self._tile("idx", [8, 1], DataType.INT32, [5, 1])

        result_type = tile.mgather(mem, idx).type

        assert isinstance(result_type, ir.TileType)
        assert [dim.value for dim in result_type.shape if isinstance(dim, ir.ConstInt)] == [8, 32]
        assert _valid_of(result_type) == [5, 32]

    @pytest.mark.parametrize(("coalesce", "expected"), [(0, 0), (1, 1)])
    def test_mgather_accepts_printed_integer_coalesce(self, coalesce, expected):
        span = ir.Span.unknown()
        mem = ir.Var("mem", ir.TensorType([64, 32], DataType.FP32), span)
        idx = self._tile("idx", [1, 8], DataType.INT32)

        call = tile.mgather(mem, idx, coalesce=coalesce)

        assert dict(call.kwargs) == {"coalesce": expected}

    def test_mgather_rejects_invalid_coalesce(self):
        span = ir.Span.unknown()
        mem = ir.Var("mem", ir.TensorType([64, 32], DataType.FP32), span)
        idx = self._tile("idx", [1, 8], DataType.INT32)
        with pytest.raises(ValueError, match="coalesce"):
            tile.mgather(mem, idx, coalesce="invalid")
        with pytest.raises(ValueError, match="coalesce"):
            tile.mgather(mem, idx, coalesce=2)
        with pytest.raises(ValueError, match="coalesce"):
            tile.mgather(mem, idx, coalesce=True)

    @pytest.mark.parametrize(("gather_oob", "expected"), [("clamp", 1), ("wrap", 2), ("zero", 3), (2, 2)])
    def test_mgather_accepts_out_of_bounds_modes(self, gather_oob, expected):
        span = ir.Span.unknown()
        mem = ir.Var("mem", ir.TensorType([64, 32], DataType.FP32), span)
        idx = self._tile("idx", [1, 8], DataType.INT32)

        call = tile.mgather(mem, idx, gather_oob=gather_oob)

        assert dict(call.kwargs) == {"coalesce": 0, "gather_oob": expected}

    @pytest.mark.parametrize("gather_oob", ["invalid", 4, True])
    def test_mgather_rejects_invalid_out_of_bounds_mode(self, gather_oob):
        span = ir.Span.unknown()
        mem = ir.Var("mem", ir.TensorType([64, 32], DataType.FP32), span)
        idx = self._tile("idx", [1, 8], DataType.INT32)

        with pytest.raises(ValueError, match="gather_oob"):
            tile.mgather(mem, idx, gather_oob=gather_oob)

    def test_mgather_rejects_dynamic_physical_shapes_required_to_be_static(self):
        span = ir.Span.unknown()
        dynamic_dim = ir.Var("dynamic_dim", ir.ScalarType(DataType.INDEX), span)
        one = ir.ConstInt(1, DataType.INDEX, span)
        thirty_two = ir.ConstInt(32, DataType.INDEX, span)
        row_mem = ir.Var("row_mem", ir.TensorType([64, 32], DataType.FP16), span)
        vec_idx = ir.Var("vec_idx", ir.TileType([one, dynamic_dim], DataType.INT32), span)
        mat_row_idx = ir.Var("mat_row_idx", ir.TensorType([one, dynamic_dim], DataType.INT32), span)
        elem_mem = ir.Var("elem_mem", ir.TensorType([512], DataType.FP16), span)
        mat_elem_idx = ir.Var("mat_elem_idx", ir.TensorType([dynamic_dim, thirty_two], DataType.INT32), span)
        scratch = ir.Var("scratch", ir.TensorType([512], DataType.FP16), span)

        with pytest.raises(ValueError, match=r"static \[1, R\] or \[R, 1\]"):
            tile.mgather(row_mem, vec_idx)
        with pytest.raises(ValueError, match="Mat output shape must be static"):
            tile.mgather(row_mem, mat_row_idx, target_memory=ir.MemorySpace.Mat)
        with pytest.raises(ValueError, match="Mat elem output shape must be static"):
            tile.mgather(
                elem_mem,
                mat_elem_idx,
                coalesce="elem",
                target_memory=ir.MemorySpace.Mat,
                scratch=scratch,
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
