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
from pypto.pypto_core import DataType


class TestQuantMxTypes:
    def test_public_quant_mx_rejects_unsupported_dtype(self):
        span = ir.Span.unknown()
        src = pl.Tile(expr=ir.Var("src", ir.TileType([16, 64], DataType.FP32), span))

        with pytest.raises(ValueError, match="supports only FP8E4M3FN"):
            pl.quant_mx(src, layout=pl.MX_A_ZZ, dtype=pl.FP8E5M2)

    def test_public_quant_mx_rejects_nd_layout(self):
        span = ir.Span.unknown()
        src = pl.Tile(expr=ir.Var("src", ir.TileType([32, 128], DataType.FP32), span))

        with pytest.raises(ValueError, match="MX_A_ZZ or TensorLayout.MX_B_NN"):
            pl.quant_mx(src, layout=pl.ND)

    def test_public_quant_mx_requires_layout(self):
        span = ir.Span.unknown()
        src = pl.Tile(expr=ir.Var("src", ir.TileType([32, 128], DataType.FP32), span))

        with pytest.raises(TypeError):
            pl.quant_mx(src)  # type: ignore[call-arg]

    def test_tquant_mx_returns_public_pair(self):
        span = ir.Span.unknown()
        src = ir.Var("src", ir.TileType([16, 64], DataType.FP32), span)

        call = ir.op.tile.tquant_mx(src, mode="mxfp8_e4m3", span=span)

        assert isinstance(call.type, ir.TupleType)
        assert len(call.type.types) == 2
        dst, scale = call.type.types
        assert isinstance(dst, ir.TileType)
        assert isinstance(scale, ir.TileType)
        assert dst.dtype == DataType.FP8E4M3FN
        assert scale.dtype == DataType.FP8E8M0
        assert isinstance(scale.shape[0], ir.ConstInt) and scale.shape[0].value == 1
        assert isinstance(scale.shape[1], ir.ConstInt) and scale.shape[1].value == 32

    def test_tquant_mx_dps_is_side_effect_only(self):
        span = ir.Span.unknown()
        src = ir.Var("src", ir.TileType([16, 64], DataType.FP32), span)
        max_scratch = ir.Var("max", ir.TileType([1, 32], DataType.FP32), span)
        scaling_scratch = ir.Var("scaling", ir.TileType([1, 32], DataType.FP32), span)
        dst = ir.Var("dst", ir.TileType([16, 64], DataType.INT8), span)
        exp = ir.Var("exp", ir.TileType([1, 32], DataType.UINT8), span)

        call = ir.op.tile.tquant_mx_dps(src, max_scratch, scaling_scratch, dst, exp, span=span)

        assert isinstance(call.type, ir.UnknownType)

    def test_tquant_mx_dps_rejects_wrong_scratch_element_count(self):
        span = ir.Span.unknown()
        src = ir.Var("src", ir.TileType([16, 64], DataType.FP32), span)
        max_scratch = ir.Var("max", ir.TileType([1, 31], DataType.FP32), span)
        scaling_scratch = ir.Var("scaling", ir.TileType([1, 32], DataType.FP32), span)
        dst = ir.Var("dst", ir.TileType([16, 64], DataType.INT8), span)
        exp = ir.Var("exp", ir.TileType([1, 32], DataType.UINT8), span)

        with pytest.raises(ValueError, match="max scratch valid element count 32"):
            ir.op.tile.tquant_mx_dps(src, max_scratch, scaling_scratch, dst, exp, span=span)

    def test_tquant_mx_rejects_dynamic_shape_before_lowering(self):
        span = ir.Span.unknown()
        m = ir.Var("m", ir.ScalarType(DataType.INDEX), span)
        k = ir.ConstInt(64, DataType.INDEX, span)
        src = ir.Var("src", ir.TileType([m, k], DataType.FP32), span)

        with pytest.raises(ValueError, match="requires static M and K"):
            ir.op.tile.tquant_mx(src, span=span)

    def test_tquant_mx_rejects_partial_valid_shape(self):
        span = ir.Span.unknown()
        src = ir.Var(
            "src",
            ir.TileType(
                [16, 64],
                DataType.FP32,
                tile_view=ir.TileView(valid_shape=[16, 32]),
            ),
            span,
        )

        with pytest.raises(ValueError, match="does not support a partial src valid_shape"):
            ir.op.tile.tquant_mx(src, span=span)

    @pytest.mark.parametrize("shape", [[8, 32], [15, 64], [16, 3744]])
    def test_tquant_mx_rejects_isa_shape_constraints(self, shape):
        span = ir.Span.unknown()
        src = ir.Var("src", ir.TileType(shape, DataType.FP32), span)

        with pytest.raises(ValueError, match=r"M divisible by 16|M\*K <= 59461"):
            ir.op.tile.tquant_mx(src, span=span)

    def test_tquant_mx_rejects_unsupported_e5m2_mode(self):
        span = ir.Span.unknown()
        src = ir.Var("src", ir.TileType([16, 64], DataType.FP32), span)

        with pytest.raises(ValueError, match="unknown mode"):
            ir.op.tile.tquant_mx(src, mode="mxfp8_e5m2", span=span)

    def test_tquant_mx_rejects_unsupported_mxfp4_mode(self):
        span = ir.Span.unknown()
        src = ir.Var("src", ir.TileType([16, 64], DataType.FP32), span)

        with pytest.raises(ValueError, match="unknown mode"):
            ir.op.tile.tquant_mx(src, mode="mxfp4", span=span)


class TestTDequantTypes:
    def test_tdequant_type(self):
        span = ir.Span.unknown()
        src = ir.Var(
            "src",
            ir.TileType(
                [16, 64],
                DataType.INT8,
                tile_view=ir.TileView(blayout=ir.TileLayout.col_major),
            ),
            span,
        )
        scale = ir.Var("scale", ir.TileType([16, 1], DataType.FP32), span)
        offset = ir.Var("offset", ir.TileType([16, 1], DataType.FP32), span)

        call = ir.op.tile.tdequant(src, scale, offset, span)

        assert isinstance(call.type, ir.TileType)
        # The canonical representation elides the implicit Vec row-major view.
        assert call.type.tile_view is None
        assert call.type.dtype == DataType.FP32

    def test_tdequant_rejects_non_fp32_parameters(self):
        span = ir.Span.unknown()
        src = ir.Var("src", ir.TileType([16, 64], DataType.INT8), span)
        scale = ir.Var("scale", ir.TileType([16, 1], DataType.FP16), span)
        offset = ir.Var("offset", ir.TileType([16, 1], DataType.FP32), span)

        with pytest.raises(ValueError, match="requires scale dtype FP32"):
            ir.op.tile.tdequant(src, scale, offset, span)

    def test_tdequant_rejects_non_row_parameter_shape(self):
        span = ir.Span.unknown()
        src = ir.Var("src", ir.TileType([16, 64], DataType.INT8), span)
        scale = ir.Var("scale", ir.TileType([16, 2], DataType.FP32), span)
        offset = ir.Var("offset", ir.TileType([16, 1], DataType.FP32), span)

        with pytest.raises(ValueError, match=r"scale shape \[rows, 1\]"):
            ir.op.tile.tdequant(src, scale, offset, span)

    def test_tdequant_rejects_different_parameter_physical_shapes(self):
        span = ir.Span.unknown()
        src = ir.Var("src", ir.TileType([16, 64], DataType.INT8), span)
        scale = ir.Var("scale", ir.TileType([16, 1], DataType.FP32), span)
        offset = ir.Var(
            "offset",
            ir.TileType(
                [16, 2],
                DataType.FP32,
                tile_view=ir.TileView(valid_shape=[16, 1]),
            ),
            span,
        )

        with pytest.raises(ValueError, match="same physical shape"):
            ir.op.tile.tdequant(src, scale, offset, span)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
