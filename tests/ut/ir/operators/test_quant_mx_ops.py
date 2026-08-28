# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for MX quantization operators."""

import pypto.language as pl
import pytest
from pypto import ir


def _tile(name, shape, dtype, *, valid_shape=None, view=None):
    if view is None and valid_shape is not None:
        view = ir.TileView(valid_shape=valid_shape)
    if any(isinstance(dim, ir.Expr) for dim in shape):
        shape = [
            dim if isinstance(dim, ir.Expr) else ir.ConstInt(dim, pl.INDEX, ir.Span.unknown())
            for dim in shape
        ]
    return ir.Var(name, ir.TileType(shape, dtype, tile_view=view), ir.Span.unknown())


def _shape_values(tile_type):
    return tuple(dim.value for dim in tile_type.shape)


class TestQuantMxTypes:
    @pytest.mark.parametrize(
        ("group_axis", "dtype", "src_shape", "quant_shape", "scale_shape"),
        [
            (1, pl.FP8E4M3FN, (16, 64), (16, 64), (16, 2)),
            (0, pl.FP8E4M3FN, (32, 64), (64, 32), (2, 32)),
        ],
    )
    def test_public_result_types(self, group_axis, dtype, src_shape, quant_shape, scale_shape):
        src = _tile("src", src_shape, pl.BF16)

        call = ir.op.tile.tquant_mx(src, group_axis=group_axis, dtype=dtype)

        assert isinstance(call.type, ir.TupleType)
        quant, scale = call.type.types
        assert isinstance(quant, ir.TileType) and quant.dtype == dtype
        assert isinstance(scale, ir.TileType) and scale.dtype == pl.FP8E8M0
        assert _shape_values(quant) == quant_shape
        assert _shape_values(scale) == scale_shape
        assert scale.tile_view is not None
        expected_layout = ir.TileLayout.col_major if group_axis == 0 else ir.TileLayout.row_major
        assert scale.tile_view.blayout == expected_layout
        assert scale.tile_view.slayout == expected_layout
        assert scale.tile_view.fractal == 32

    def test_public_quantized_view_matches_lowered_destination(self):
        src = _tile(
            "src",
            (32, 64),
            pl.BF16,
            view=ir.TileView(
                blayout=ir.TileLayout.col_major,
                slayout=ir.TileLayout.col_major,
            ),
        )

        result_type = ir.op.tile.tquant_mx(src, group_axis=0).type
        assert isinstance(result_type, ir.TupleType)
        quant, _scale = result_type.types

        assert isinstance(quant, ir.TileType)
        view = quant.get_effective_tile_view()
        assert view.blayout == ir.TileLayout.row_major
        assert view.slayout == ir.TileLayout.none_box

    def test_dtype_kwarg_round_trips_through_python_printer(self):
        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(self, src: pl.Tensor[[16, 64], pl.FP16]):
                _quant, _scale = pl.quant_mx(
                    pl.load(src, [0, 0], [16, 64]),
                    group_axis=1,
                    dtype=pl.FP8E4M3FN,
                )

        reparsed = pl.parse_program(str(Program))
        ir.assert_structural_equal(reparsed, Program)

    def test_public_rejects_missing_or_invalid_group_axis(self):
        src = pl.Tile(expr=_tile("src", (16, 64), pl.FP16))
        with pytest.raises(TypeError):
            pl.quant_mx(src)  # type: ignore[call-arg]
        with pytest.raises(ValueError, match="group_axis must be 0 or 1"):
            pl.quant_mx(src, group_axis=2)

    def test_public_rejects_unsupported_dtype(self):
        src = pl.Tile(expr=_tile("src", (16, 64), pl.FP16))
        with pytest.raises(ValueError, match="supports only FP8E4M3FN"):
            pl.quant_mx(src, group_axis=1, dtype=pl.FP8E5M2)
        with pytest.raises(ValueError, match="supports only FP8E4M3FN"):
            pl.quant_mx(src, group_axis=1, dtype=pl.FP4)

    @pytest.mark.parametrize(
        ("shape", "dtype", "group_axis", "message"),
        [
            ((16, 64), pl.INT8, 1, "requires src dtype in"),
            ((16, 64), pl.FP16, 0, "N divisible by 32"),
            ((16, 32), pl.FP16, 1, "K divisible by 64"),
        ],
    )
    def test_rejects_dtype_and_alignment_constraints(self, shape, dtype, group_axis, message):
        with pytest.raises(ValueError, match=message):
            ir.op.tile.tquant_mx(_tile("src", shape, dtype), group_axis=group_axis, dtype=pl.FP8E4M3FN)

    def test_rejects_dynamic_or_partial_source(self):
        dim = ir.Var("m", ir.ScalarType(pl.INDEX), ir.Span.unknown())
        with pytest.raises(ValueError, match="requires static M and K"):
            ir.op.tile.tquant_mx(_tile("src", (dim, 64), pl.FP16), group_axis=1)
        with pytest.raises(ValueError, match="partial src valid_shape"):
            ir.op.tile.tquant_mx(_tile("src", (16, 64), pl.FP16, valid_shape=[16, 32]), group_axis=1)

    def test_raw_is_value_returning(self):
        src = _tile("src", (16, 64), pl.FP16)
        call = ir.op.tile.tquant_mx_raw(
            src,
            _tile("max", (1, 32), pl.FP16),
            _tile("scaling", (1, 32), pl.FP16),
            dtype=pl.FP8E4M3FN,
            group_axis=1,
        )
        assert isinstance(call.type, ir.TupleType)
        dst, exp = call.type.types
        assert isinstance(dst, ir.TileType) and dst.dtype == pl.INT8
        assert isinstance(exp, ir.TileType) and exp.dtype == pl.UINT8

    def test_raw_rejects_wrong_scratch_size(self):
        with pytest.raises(ValueError, match="max scratch valid element count 32"):
            ir.op.tile.tquant_mx_raw(
                _tile("src", (16, 64), pl.FP16),
                _tile("max", (1, 31), pl.FP16),
                _tile("scaling", (1, 32), pl.FP16),
                group_axis=1,
            )

    def test_tmov_x2zz_value_return_and_axis0_tmp(self):
        # Axis1: legacy-flat src [1,32] -> ZZ [16,2]; tmp covers 64+ceil(16/16)*2.
        src = _tile("src", (1, 32), pl.UINT8)
        tmp = _tile("tmp", (1, 96), pl.UINT8)
        call = ir.op.tile.tmov_x2zz(src, tmp, group_axis=1, dst_rows=16, dst_cols=2)
        assert isinstance(call.type, ir.TileType)
        assert call.type.dtype == pl.UINT8
        assert _shape_values(call.type) == (16, 2)

        # Axis0: DN [2,16] -> ZZ [16,2]; minimal 32-byte Vec pad is required.
        src0 = _tile("src0", (2, 16), pl.UINT8)
        tmp0 = _tile("tmp0", (1, 32), pl.UINT8)
        call0 = ir.op.tile.tmov_x2zz(src0, tmp0, group_axis=0)
        assert _shape_values(call0.type) == (16, 2)
        with pytest.raises(ValueError, match="tmp capacity of at least 32"):
            ir.op.tile.tmov_x2zz(src0, _tile("tiny", (1, 16), pl.UINT8), group_axis=0)
        with pytest.raises(ValueError, match="requires dst_rows and dst_cols"):
            ir.op.tile.tmov_x2zz(src, tmp, group_axis=1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
