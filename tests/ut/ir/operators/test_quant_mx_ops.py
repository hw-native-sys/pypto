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


def _const_int(expr: ir.Expr) -> int:
    assert isinstance(expr, ir.ConstInt)
    return expr.value


def _shape_values(tile_type):
    return tuple(dim.value for dim in tile_type.shape)


class TestQuantMxTypes:
    @pytest.mark.parametrize(
        ("group_axis", "dtype", "src_shape", "quant_shape", "scale_shape"),
        [
            (1, pl.FP8E4M3FN, (16, 64), (16, 64), (16, 2)),
            (0, pl.FP8E4M3FN, (32, 64), (64, 32), (2, 32)),
            (1, pl.FP4E2M1X2, (16, 64), (16, 32), (16, 2)),
            (0, pl.FP4E2M1X2, (64, 64), (64, 32), (2, 64)),
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

    @pytest.mark.parametrize("dtype", [pl.FP8E5M2, pl.FP4E1M2X2])
    def test_public_rejects_unsupported_dtype(self, dtype):
        src = pl.Tile(expr=_tile("src", (16, 64), pl.FP16))
        with pytest.raises(ValueError, match="requires dtype FP8E4M3FN or FP4E2M1X2"):
            pl.quant_mx(src, group_axis=1, dtype=dtype)

    @pytest.mark.parametrize(
        ("source_dtype", "shape", "axis", "algorithm", "message"),
        [
            (pl.FP32, (16, 64), 1, "ocp", "requires FP16 or BF16 source"),
            (pl.FP16, (32, 64), 0, "ocp", "N divisible by 64"),
            (pl.BF16, (16, 64), 1, "nv", "axis1 FP16/BF16 NV"),
            (pl.BF16, (64, 64), 0, "invalid", "quant_scale_alg 'ocp' or 'nv'"),
        ],
    )
    def test_fp4_rejects_unsupported_source_or_algorithm(self, source_dtype, shape, axis, algorithm, message):
        with pytest.raises(ValueError, match=message):
            ir.op.tile.tquant_mx(
                _tile("src", shape, source_dtype),
                group_axis=axis,
                dtype=pl.FP4E2M1X2,
                quant_scale_alg=algorithm,
            )

    def test_fp4_raw_keeps_native_carrier_type(self):
        # Raw FP4 TQUANT writes native byte carriers; FP8's INT8 alias is inapplicable.
        call = ir.op.tile.tquant_mx_raw(
            _tile("src", (64, 64), pl.BF16),
            _tile("max", (2, 64), pl.BF16),
            _tile("scaling", (2, 64), pl.BF16),
            group_axis=0,
            dtype=pl.FP4E2M1X2,
            quant_scale_alg="nv",
        )
        assert isinstance(call.type, ir.TupleType)
        dst, exp = call.type.types
        assert isinstance(dst, ir.TileType) and dst.dtype == pl.FP4E2M1X2
        assert _shape_values(dst) == (64, 32)
        assert _shape_values(exp) == (2, 64)

    @pytest.mark.parametrize("rows", [1, 3])
    def test_fp4_raw_axis1_accepts_small_rows(self, rows):
        quant, exp = pl.tile.quant_mx_raw(
            pl.Tile(expr=_tile("src", (4, 64), pl.BF16, valid_shape=[rows, 64])),
            pl.Tile(expr=_tile("max", (1, 16), pl.BF16, valid_shape=[1, rows * 2])),
            pl.Tile(expr=_tile("scaling", (1, 128), pl.BF16, valid_shape=[1, rows * 2])),
            group_axis=1,
            dtype=pl.FP4E2M1X2,
        )
        quant_type = quant.unwrap().type
        assert isinstance(quant_type, ir.TileType)
        assert quant_type.tile_view is not None
        assert _shape_values(quant_type) == (4, 32)
        assert tuple(_const_int(dim) for dim in quant_type.tile_view.valid_shape) == (rows, 32)
        exp_type = exp.unwrap().type
        assert isinstance(exp_type, ir.TileType) and exp_type.tile_view is not None
        assert _shape_values(exp_type) == (1, 32)
        assert tuple(_const_int(dim) for dim in exp_type.tile_view.valid_shape) == (1, rows * 2)

    def test_fp4_raw_axis0_preserves_padded_row_stride(self):
        call = ir.op.tile.tquant_mx_raw(
            _tile("src", (64, 64), pl.BF16, valid_shape=[32, 32]),
            _tile("max", (1, 64), pl.BF16, valid_shape=[1, 32]),
            _tile("scaling", (1, 64), pl.BF16, valid_shape=[1, 32]),
            group_axis=0,
            dtype=pl.FP4E2M1X2,
        )
        assert isinstance(call.type, ir.TupleType)
        quant, exp = call.type.types
        assert isinstance(quant, ir.TileType) and quant.tile_view is not None
        assert isinstance(exp, ir.TileType) and exp.tile_view is not None
        assert _shape_values(quant) == (64, 32)
        assert tuple(_const_int(dim) for dim in quant.tile_view.valid_shape) == (32, 16)
        assert _shape_values(exp) == (1, 64)
        assert tuple(_const_int(dim) for dim in exp.tile_view.valid_shape) == (1, 32)

    def test_fp4_raw_interleaved_exponent_shape(self):
        call = ir.op.tile.tquant_mx_raw(
            _tile("src", (128, 64), pl.BF16),
            _tile("max", (4, 64), pl.BF16),
            _tile("scaling", (4, 64), pl.BF16),
            group_axis=0,
            dtype=pl.FP4E2M1X2,
            interleave=True,
        )
        assert isinstance(call.type, ir.TupleType)
        assert _shape_values(call.type.types[1]) == (2, 128)

    def test_fp4_raw_interleave_rejects_one_row_group(self):
        with pytest.raises(ValueError, match="physical and valid rows divisible by 64"):
            ir.op.tile.tquant_mx_raw(
                _tile("src", (32, 64), pl.BF16),
                _tile("max", (1, 64), pl.BF16),
                _tile("scaling", (1, 64), pl.BF16),
                group_axis=0,
                dtype=pl.FP4E2M1X2,
                interleave=True,
            )

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


class TestTensorQuantMxTypes:
    @pytest.mark.parametrize(
        ("axis", "quant_shape", "scale_shape"), [(1, (64, 64), (64, 4)), (0, (128, 32), (4, 64))]
    )
    def test_fp4_tensor_carrier_shapes(self, axis, quant_shape, scale_shape):
        src = ir.Var("src", ir.TensorType([64, 128], pl.BF16), ir.Span.unknown())
        call = ir.op.tensor.quant_mx(src, group_axis=axis, dtype=pl.FP4E2M1X2)
        assert isinstance(call.type, ir.TupleType)
        quant, scale = call.type.types
        assert isinstance(quant, ir.TensorType) and quant.dtype == pl.FP4E2M1X2
        assert _shape_values(quant) == quant_shape
        assert _shape_values(scale) == scale_shape

    @pytest.mark.parametrize(
        ("group_axis", "src_shape", "quant_shape", "scale_shape", "layout"),
        [
            (1, (16, 64), (16, 64), (16, 2), ir.TensorLayout.MX_A_ZZ),
            (0, (32, 64), (64, 32), (2, 32), ir.TensorLayout.MX_B_NN),
        ],
    )
    def test_tensor_quant_mx_returns_oriented_data_and_mx_scale(
        self, group_axis, src_shape, quant_shape, scale_shape, layout
    ):
        span = ir.Span.unknown()
        src = ir.Var("src", ir.TensorType(src_shape, pl.BF16), span)

        call = ir.op.tensor.quant_mx(src, group_axis=group_axis)

        assert isinstance(call.type, ir.TupleType)
        quant, scale = call.type.types
        assert isinstance(quant, ir.TensorType) and quant.dtype == pl.FP8E4M3FN
        assert isinstance(scale, ir.TensorType) and scale.dtype == pl.FP8E8M0
        assert _shape_values(quant) == quant_shape
        assert _shape_values(scale) == scale_shape
        assert scale.tensor_view is not None and scale.tensor_view.layout == layout

    def test_tensor_quant_mx_rejects_misaligned_shapes(self):
        span = ir.Span.unknown()
        with pytest.raises(ValueError, match="N divisible by 32"):
            ir.op.tensor.quant_mx(ir.Var("src", ir.TensorType([16, 64], pl.FP16), span), group_axis=0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
