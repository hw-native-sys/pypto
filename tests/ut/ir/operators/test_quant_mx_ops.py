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
        ("layout", "dtype", "src_shape", "quant_shape", "scale_shape"),
        [
            (pl.MX_A_ZZ, pl.FP8E4M3FN, (16, 64), (16, 64), (16, 2)),
            (pl.MX_A_ZZ, pl.FP4, (16, 64), (16, 64), (16, 2)),
            (pl.MX_B_NN, pl.FP8E4M3FN, (32, 64), (64, 32), (2, 32)),
            (pl.MX_B_NN, pl.FP4, (64, 64), (64, 64), (2, 64)),
        ],
    )
    def test_public_result_types(self, layout, dtype, src_shape, quant_shape, scale_shape):
        src = _tile("src", src_shape, pl.BF16)

        call = ir.op.tile.tquant_mx(src, layout=layout, dtype=dtype)

        assert isinstance(call.type, ir.TupleType)
        quant, scale = call.type.types
        assert isinstance(quant, ir.TileType) and quant.dtype == dtype
        assert isinstance(scale, ir.TileType) and scale.dtype == pl.FP8E8M0
        assert _shape_values(quant) == quant_shape
        assert _shape_values(scale) == scale_shape
        assert scale.tile_view is not None
        expected_layout = ir.TileLayout.col_major if layout == pl.MX_B_NN else ir.TileLayout.row_major
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

        result_type = ir.op.tile.tquant_mx(src, layout=pl.MX_B_NN).type
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
                    layout=pl.MX_A_ZZ,
                    dtype=pl.FP4,
                )

        reparsed = pl.parse_program(str(Program))
        ir.assert_structural_equal(reparsed, Program)

    @pytest.mark.parametrize("layout", [pl.ND, None])
    def test_public_rejects_non_mx_layout(self, layout):
        src = pl.Tile(expr=_tile("src", (16, 64), pl.FP16))
        if layout is None:
            with pytest.raises(TypeError):
                pl.quant_mx(src)  # type: ignore[call-arg]
        else:
            with pytest.raises(ValueError, match=r"MX_A_ZZ or TensorLayout\.MX_B_NN"):
                pl.quant_mx(src, layout=layout)

    def test_public_rejects_unsupported_dtype(self):
        src = pl.Tile(expr=_tile("src", (16, 64), pl.FP16))
        with pytest.raises(ValueError, match="supports FP8E4M3FN or FP4"):
            pl.quant_mx(src, layout=pl.MX_A_ZZ, dtype=pl.FP8E5M2)

    @pytest.mark.parametrize(
        ("shape", "dtype", "layout", "message"),
        [
            ((16, 64), pl.FP32, pl.MX_A_ZZ, "requires src dtype in"),
            ((16, 64), pl.FP16, pl.MX_B_NN, "N divisible by 32"),
            ((32, 64), pl.FP16, pl.MX_B_NN, "FP16 MXFP4 requires N divisible by 64"),
            ((16, 32), pl.FP16, pl.MX_A_ZZ, "K divisible by 64"),
        ],
    )
    def test_rejects_dtype_and_alignment_constraints(self, shape, dtype, layout, message):
        quant_dtype = pl.FP4 if "src dtype" in message or "MXFP4" in message else pl.FP8E4M3FN
        with pytest.raises(ValueError, match=message):
            ir.op.tile.tquant_mx(_tile("src", shape, dtype), layout=layout, dtype=quant_dtype)

    def test_rejects_dynamic_or_partial_source(self):
        dim = ir.Var("m", ir.ScalarType(pl.INDEX), ir.Span.unknown())
        with pytest.raises(ValueError, match="requires static M and K"):
            ir.op.tile.tquant_mx(_tile("src", (dim, 64), pl.FP16), layout=pl.MX_A_ZZ)
        with pytest.raises(ValueError, match="partial src valid_shape"):
            ir.op.tile.tquant_mx(_tile("src", (16, 64), pl.FP16, valid_shape=[16, 32]), layout=pl.MX_A_ZZ)

    @pytest.mark.parametrize(("dtype", "dst_dtype"), [(pl.FP8E4M3FN, pl.INT8), (pl.FP4, pl.FP4)])
    def test_dps_is_side_effect_only(self, dtype, dst_dtype):
        src = _tile("src", (16, 64), pl.FP16)
        call = ir.op.tile.tquant_mx_dps(
            src,
            _tile("max", (1, 32), pl.FP16),
            _tile("scaling", (1, 32), pl.FP16),
            _tile("dst", (16, 64), dst_dtype),
            _tile("exp", (1, 32), pl.UINT8),
            dtype=dtype,
        )
        assert isinstance(call.type, ir.UnknownType)

    def test_dps_rejects_wrong_scratch_size(self):
        with pytest.raises(ValueError, match="max scratch valid element count 32"):
            ir.op.tile.tquant_mx_dps(
                _tile("src", (16, 64), pl.FP16),
                _tile("max", (1, 31), pl.FP16),
                _tile("scaling", (1, 32), pl.FP16),
                _tile("dst", (16, 64), pl.INT8),
                _tile("exp", (1, 32), pl.UINT8),
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
