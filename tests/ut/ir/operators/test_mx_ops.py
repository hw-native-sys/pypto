# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for MX scale layouts, matmul_mx, and internal scale-address binding."""

import pypto.language as pl
import pytest
from pypto import ir
from pypto.pypto_core import DataType


class TestMxLoad:
    def test_mx_layout_sets_fractal(self):
        span = ir.Span.unknown()
        tensor = ir.Var(
            "t",
            ir.TensorType(
                [16, 2],
                DataType.FP8E8M0,
                tensor_view=ir.TensorView([], ir.TensorLayout.MX_A_ZZ),
            ),
            span,
        )
        call = ir.op.tile.load(
            tensor,
            [0, 0],
            [16, 2],
            target_memory=ir.MemorySpace.Mat,
            span=span,
        )
        tile_type = call.type
        assert isinstance(tile_type, ir.TileType)
        assert tile_type.dtype == DataType.FP8E8M0
        assert tile_type.tile_view is not None
        assert tile_type.tile_view.fractal == 32

    def test_regular_nd_load_does_not_select_mx_layout(self):
        span = ir.Span.unknown()
        tensor = ir.Var("t", ir.TensorType([16, 2], DataType.FP8E8M0), span)
        call = ir.op.tile.load(tensor, [0, 0], [16, 2], target_memory=ir.MemorySpace.Mat, span=span)
        tile_type = call.type
        assert isinstance(tile_type, ir.TileType)
        assert tile_type.tile_view is None or tile_type.tile_view.fractal != 32

    def test_rejects_vec_target_with_mx_layout(self):
        span = ir.Span.unknown()
        tensor = ir.Var(
            "t",
            ir.TensorType(
                [16, 2],
                DataType.FP8E8M0,
                tensor_view=ir.TensorView([], ir.TensorLayout.MX_A_ZZ),
            ),
            span,
        )
        with pytest.raises(ValueError, match="Mat|Vec"):
            ir.op.tile.load(
                tensor,
                [0, 0],
                [16, 2],
                target_memory=ir.MemorySpace.Vec,
                span=span,
            )

    def test_mx_layout_without_target_memory_is_rejected(self):
        span = ir.Span.unknown()
        tensor = ir.Var(
            "t",
            ir.TensorType(
                [16, 2],
                DataType.FP8E8M0,
                tensor_view=ir.TensorView([], ir.TensorLayout.MX_A_ZZ),
            ),
            span,
        )
        offsets = ir.MakeTuple(
            [ir.ConstInt(0, DataType.INDEX, span), ir.ConstInt(0, DataType.INDEX, span)], span
        )
        shapes = ir.MakeTuple(
            [ir.ConstInt(16, DataType.INDEX, span), ir.ConstInt(2, DataType.INDEX, span)], span
        )
        with pytest.raises(ValueError, match="requires target_memory=MemorySpace.Mat"):
            ir.create_op_call(
                "tile.load",
                [tensor, offsets, shapes, shapes],
                {},
                span,
            )


class TestDtypeAndMemorySpace:
    def test_fp8e8m0_exists(self):
        assert DataType.FP8E8M0.get_bit() == 8
        assert DataType.FP8E8M0.to_string() == "fp8e8m0"
        assert pl.FP8E8M0 == DataType.FP8E8M0

    def test_left_right_scale_spaces(self):
        assert ir.MemorySpace.LeftScale == pl.Mem.LeftScale
        assert ir.MemorySpace.RightScale == pl.Mem.RightScale

    def test_memory_space_serialized_values_are_stable(self):
        assert ir.MemorySpace.ScalarLocal.value == 7
        assert ir.MemorySpace.LeftScale.value == 8
        assert ir.MemorySpace.RightScale.value == 9


class TestTensorMatmulMx:
    def test_tensor_matmul_mx_requires_oriented_scale_layouts(self):
        span = ir.Span.unknown()
        lhs = ir.Var("lhs", ir.TensorType([16, 64], pl.FP8E4M3FN), span)
        lhs_scale = ir.Var(
            "lhs_scale",
            ir.TensorType([16, 2], pl.FP8E8M0, tensor_view=ir.TensorView([], ir.TensorLayout.MX_A_ZZ)),
            span,
        )
        rhs = ir.Var("rhs", ir.TensorType([64, 32], pl.FP8E4M3FN), span)
        rhs_scale = ir.Var(
            "rhs_scale",
            ir.TensorType([2, 32], pl.FP8E8M0, tensor_view=ir.TensorView([], ir.TensorLayout.MX_B_NN)),
            span,
        )

        call = ir.op.tensor.matmul_mx(lhs, lhs_scale, rhs, rhs_scale)
        assert isinstance(call.type, ir.TensorType)
        assert call.type.dtype == pl.FP32
        assert tuple(dim.value for dim in call.type.shape if isinstance(dim, ir.ConstInt)) == (16, 32)

        bad_scale = ir.Var("bad_scale", ir.TensorType([16, 2], pl.FP8E8M0), span)
        with pytest.raises(ValueError, match="lhs_scale layout MX_A_ZZ"):
            ir.op.tensor.matmul_mx(lhs, bad_scale, rhs, rhs_scale)


class TestPackedFp4Shape:
    @pytest.mark.parametrize("dtype", [DataType.FP4E2M1X2, DataType.FP4E1M2X2])
    @pytest.mark.parametrize("type_constructor", [ir.TensorType, ir.TileType, ir.DistributedTensorType])
    def test_odd_carrier_dimensions_preserve_their_units(self, type_constructor, dtype):
        shaped_type = type_constructor([8, 15], dtype)
        assert shaped_type.dtype == dtype
        assert [dim.value for dim in shaped_type.shape if isinstance(dim, ir.ConstInt)] == [8, 15]

    @pytest.mark.parametrize("dtype", [DataType.FP4E2M1X2, DataType.FP4E1M2X2])
    def test_accepts_symbolic_carrier_last_dimension(self, dtype):
        span = ir.Span.unknown()
        carrier_width = ir.Var("carrier_width", ir.ScalarType(DataType.INDEX), span)
        tensor_type = ir.TensorType([ir.ConstInt(8, DataType.INDEX, span), carrier_width], dtype)
        assert tensor_type.dtype == dtype
        assert str(tensor_type.shape[1]) == "carrier_width"


def _tile(name, shape, dtype, *, valid_shape=None, memory=None):
    span = ir.Span.unknown()
    view = None
    if valid_shape is not None:
        view = ir.TileView(valid_shape=valid_shape, stride=[])
    return ir.Var(name, ir.TileType(shape, dtype, tile_view=view, memory_space=memory), span)


class TestMatmulMxRegistry:
    def test_memory_specs(self):
        assert not hasattr(pl, "tget_scale_addr")
        mx = ir.get_op_memory_spec("tile.matmul_mx")
        assert mx is not None
        assert mx["output_memory"] == ir.MemorySpace.Acc
        assert mx["input_constraints"] == [
            [ir.MemorySpace.Left],
            [ir.MemorySpace.LeftScale],
            [ir.MemorySpace.Right],
            [ir.MemorySpace.RightScale],
        ]
        acc = ir.get_op_memory_spec("tile.matmul_mx_acc")
        assert acc is not None
        assert acc["input_constraints"][0] == [ir.MemorySpace.Acc]
        bias = ir.get_op_memory_spec("tile.matmul_mx_bias")
        assert bias is not None
        assert bias["input_constraints"][4] == [ir.MemorySpace.Bias]
        tget = ir.get_op_memory_spec("tile.tget_scale_addr")
        assert tget is not None
        assert tget["input_constraints"] == []
        assert tget["output_memory"] == "inherit_from_input"


class TestMatmulMxTypes:
    @staticmethod
    def _mx_operands(span, *, m=16, k=64, n=32, valid=None, dtype=DataType.FP8E4M3FN):
        """Build a valid MX operand set; optional valid=(m_v, k_v, n_v)."""
        if valid is None:
            lhs = ir.Var("lhs", ir.TileType([m, k], dtype), span)
            lhs_scale = ir.Var("lhs_scale", ir.TileType([m, k // 32], DataType.FP8E8M0), span)
            rhs = ir.Var("rhs", ir.TileType([k, n], dtype), span)
            rhs_scale = ir.Var("rhs_scale", ir.TileType([k // 32, n], DataType.FP8E8M0), span)
            return lhs, lhs_scale, rhs, rhs_scale
        m_v, k_v, n_v = valid
        sk = (k + 31) // 32
        sk_v = (k_v + 31) // 32
        lhs = _tile("lhs", [m, k], dtype, valid_shape=[m_v, k_v])
        lhs_scale = _tile("lhs_scale", [m, sk], DataType.FP8E8M0, valid_shape=[m_v, sk_v])
        rhs = _tile("rhs", [k, n], dtype, valid_shape=[k_v, n_v])
        rhs_scale = _tile("rhs_scale", [sk, n], DataType.FP8E8M0, valid_shape=[sk_v, n_v])
        return lhs, lhs_scale, rhs, rhs_scale

    def test_type_deduction_and_variants(self):
        span = ir.Span.unknown()
        lhs, lhs_scale, rhs, rhs_scale = self._mx_operands(span)
        call = ir.op.tile.matmul_mx(lhs, lhs_scale, rhs, rhs_scale, span)
        call_type = call.type
        assert isinstance(call_type, ir.TileType)
        assert call_type.dtype == DataType.FP32
        rows, cols = call_type.shape
        assert isinstance(rows, ir.ConstInt) and rows.value == 16
        assert isinstance(cols, ir.ConstInt) and cols.value == 32

        acc = ir.Var("acc", ir.TileType([16, 32], DataType.FP32), span)
        acc_call = ir.op.tile.matmul_mx_acc(acc, lhs, lhs_scale, rhs, rhs_scale, span)
        assert acc_call.op.name == ir.get_op("tile.matmul_mx_acc").name

        bias = ir.Var("bias", ir.TileType([1, 32], DataType.FP32), span)
        bias_call = ir.op.tile.matmul_mx_bias(lhs, lhs_scale, rhs, rhs_scale, bias, span)
        assert bias_call.op.name == ir.get_op("tile.matmul_mx_bias").name

    def test_rejects_reused_scale_between_operands(self):
        """A/B scales require distinct operand tiles and layouts."""
        span = ir.Span.unknown()
        lhs, scale, rhs, _ = self._mx_operands(span, m=32, k=1024, n=32)

        with pytest.raises(ValueError, match="requires distinct lhs_scale and rhs_scale tiles"):
            ir.op.tile.matmul_mx(lhs, scale, rhs, scale, span)

    @pytest.mark.parametrize("dtype", [DataType.FP4E2M1X2, DataType.FP4E1M2X2])
    def test_rejects_native_mxfp4_and_requires_cast(self, dtype):
        span = ir.Span.unknown()
        fp4_operands = self._mx_operands(span, n=64, dtype=dtype)
        with pytest.raises(ValueError, match=r"requires lhs dtype FP8E4M3FN or FP8E5M2.*packed FP4.*pl.cast"):
            ir.op.tile.matmul_mx(*fp4_operands, span)

        lhs, lhs_scale, _, _ = self._mx_operands(span, n=64)
        _, _, rhs_fp4, rhs_scale = self._mx_operands(span, n=64, dtype=dtype)
        with pytest.raises(ValueError, match=r"requires rhs dtype FP8E4M3FN or FP8E5M2.*packed FP4.*pl.cast"):
            ir.op.tile.matmul_mx(lhs, lhs_scale, rhs_fp4, rhs_scale, span)

    @pytest.mark.parametrize("rejected", [DataType.HF8, DataType.FP16, DataType.BF16, DataType.INT8])
    def test_rejects_non_mx_data_dtypes_per_side(self, rejected):
        """Each side is checked independently against the MX operand table.

        HF8 is the interesting one: the Cube multiplies HF8 x HF8 for ordinary
        matmul and GEMV, but PTOAS gates MX through a separate predicate
        (`isA5MxFp8InputType`) that admits only E4M3FN and E5M2, so HF8 must be
        rejected here rather than handed to PTOAS.
        """
        span = ir.Span.unknown()
        lhs, lhs_scale, rhs, rhs_scale = self._mx_operands(span)
        bad_lhs = ir.Var("bad_lhs", ir.TileType([16, 64], rejected), span)
        bad_rhs = ir.Var("bad_rhs", ir.TileType([64, 32], rejected), span)

        with pytest.raises(ValueError, match="requires lhs dtype FP8E4M3FN or FP8E5M2") as exc_info:
            ir.op.tile.matmul_mx(bad_lhs, lhs_scale, rhs, rhs_scale, span)
        assert "packed FP4" not in str(exc_info.value)

        with pytest.raises(ValueError, match="requires rhs dtype FP8E4M3FN or FP8E5M2") as exc_info:
            ir.op.tile.matmul_mx(lhs, lhs_scale, bad_rhs, rhs_scale, span)
        assert "packed FP4" not in str(exc_info.value)

    def test_rejects_wrong_dtypes_and_alignment(self):
        span = ir.Span.unknown()
        # scale must be FP8E8M0
        lhs, _, rhs, rhs_scale = self._mx_operands(span)
        with pytest.raises(ValueError, match="FP8E8M0"):
            ir.op.tile.matmul_mx(
                lhs, ir.Var("ls", ir.TileType([16, 2], DataType.FP16), span), rhs, rhs_scale, span
            )
        # M%16 / K%64 / N%32 (FP8)
        with pytest.raises(ValueError, match="divisible by 16"):
            ir.op.tile.matmul_mx(*self._mx_operands(span, m=8), span)
        with pytest.raises(ValueError, match="divisible by 64"):
            ir.op.tile.matmul_mx(*self._mx_operands(span, k=96), span)
        with pytest.raises(ValueError, match=r"physical N divisible by 32 for fp8e4m3fn"):
            ir.op.tile.matmul_mx(*self._mx_operands(span, n=16), span)

    @pytest.mark.parametrize(
        ("rhs_k", "rhs_scale_groups", "error"),
        [
            pytest.param(96, 3, "rhs K", id="misaligned-rhs-k"),
            pytest.param(0, 1, "rhs K", id="non-positive-rhs-k"),
            pytest.param(128, 2, "rhs_scale physical rows=4", id="wrong-rhs-scale-groups"),
        ],
    )
    def test_rejects_invalid_constant_rhs_k_when_lhs_k_is_symbolic(self, rhs_k, rhs_scale_groups, error):
        """Constant RHS K and scale groups are validated independently of symbolic LHS K."""
        span = ir.Span.unknown()
        k_sym = ir.Var("K", ir.ScalarType(DataType.INDEX), span)
        m16 = ir.ConstInt(16, DataType.INDEX, span)
        n32 = ir.ConstInt(32, DataType.INDEX, span)
        lhs = ir.Var("lhs", ir.TileType([m16, k_sym], DataType.FP8E4M3FN), span)
        lhs_scale = ir.Var(
            "ls", ir.TileType([m16, ir.ConstInt(2, DataType.INDEX, span)], DataType.FP8E8M0), span
        )
        rhs = ir.Var(
            "rhs",
            ir.TileType([ir.ConstInt(rhs_k, DataType.INDEX, span), n32], DataType.FP8E4M3FN),
            span,
        )
        rhs_scale = ir.Var(
            "rs",
            ir.TileType([ir.ConstInt(rhs_scale_groups, DataType.INDEX, span), n32], DataType.FP8E8M0),
            span,
        )

        with pytest.raises(ValueError, match=error):
            ir.op.tile.matmul_mx(lhs, lhs_scale, rhs, rhs_scale, span)

    def test_valid_shape_contract(self):
        span = ir.Span.unknown()
        # valid K not multiple of 32 is OK when scale-group count matches physical K
        call = ir.op.tile.matmul_mx(*self._mx_operands(span, valid=(16, 48, 32)), span)
        call_type = call.type
        assert isinstance(call_type, ir.TileType)
        assert call_type.dtype == DataType.FP32

        # propagate contracted M/N valid into Acc output
        call = ir.op.tile.matmul_mx(*self._mx_operands(span, valid=(8, 64, 16)), span)
        call_type = call.type
        assert isinstance(call_type, ir.TileType)
        valid_rows, valid_cols = call_type.get_effective_tile_view().valid_shape
        assert isinstance(valid_rows, ir.ConstInt) and valid_rows.value == 8
        assert isinstance(valid_cols, ir.ConstInt) and valid_cols.value == 16

        with pytest.raises(ValueError, match="matching valid K"):
            lhs = _tile("lhs", [16, 64], DataType.FP8E4M3FN, valid_shape=[16, 48])
            lhs_scale = _tile("ls", [16, 2], DataType.FP8E8M0)
            rhs = _tile("rhs", [64, 32], DataType.FP8E4M3FN, valid_shape=[40, 32])
            rhs_scale = _tile("rs", [2, 32], DataType.FP8E8M0)
            ir.op.tile.matmul_mx(lhs, lhs_scale, rhs, rhs_scale, span)

        with pytest.raises(ValueError, match="physical cols"):
            lhs, _, rhs, rhs_scale = self._mx_operands(span)
            ir.op.tile.matmul_mx(
                lhs, ir.Var("ls", ir.TileType([16, 1], DataType.FP8E8M0), span), rhs, rhs_scale, span
            )

        with pytest.raises(ValueError, match="valid cols"):
            lhs = _tile("lhs", [16, 64], DataType.FP8E4M3FN, valid_shape=[16, 48])
            lhs_scale = _tile("ls", [16, 2], DataType.FP8E8M0, valid_shape=[16, 1])
            rhs = _tile("rhs", [64, 32], DataType.FP8E4M3FN, valid_shape=[48, 32])
            rhs_scale = _tile("rs", [2, 32], DataType.FP8E8M0, valid_shape=[2, 32])
            ir.op.tile.matmul_mx(lhs, lhs_scale, rhs, rhs_scale, span)

        with pytest.raises(ValueError, match="positive valid K"):
            ir.op.tile.matmul_mx(*self._mx_operands(span, valid=(16, 0, 32)), span)

        # ceil(validK/32) must equal ceil(physicalK/32) (PTOAS matmul_mx vs tget)
        with pytest.raises(ValueError, match="scale-group count"):
            ir.op.tile.matmul_mx(*self._mx_operands(span, valid=(16, 31, 32)), span)

        # acc valid_shape must match MX output valid M/N
        acc = _tile("acc", [16, 32], DataType.FP32, valid_shape=[16, 32])
        with pytest.raises(ValueError, match="acc valid rows"):
            ir.op.tile.matmul_mx_acc(acc, *self._mx_operands(span, valid=(8, 64, 16)), span)

    def test_narrowed_rows_stamp_compact_and_full_rows_do_not(self):
        """Issue #2470: MX shares the L0C stride contract of the plain matmuls.

        ``mad`` takes M from the lhs *valid* rows and lays L0C out at
        ceil(M/16)*16, while the Acc readers key off the physical ``Rows``
        unless the tile is compact.  All three MX ops route their output type
        through the same deducer, so ``matmul_mx`` covers the family.
        """
        span = ir.Span.unknown()

        narrowed = ir.op.tile.matmul_mx(*self._mx_operands(span, valid=(8, 64, 16)), span).type
        assert isinstance(narrowed, ir.TileType)
        assert narrowed.get_effective_tile_view().compact == ir.CompactMode.normal

        full = ir.op.tile.matmul_mx(*self._mx_operands(span, valid=(16, 64, 32)), span).type
        assert isinstance(full, ir.TileType)
        assert full.get_effective_tile_view().compact == ir.CompactMode.null

    @pytest.mark.parametrize(
        ("index", "name", "shape", "dtype", "valid_shape"),
        [
            pytest.param(0, "lhs", [16, 64], DataType.FP8E4M3FN, [16], id="lhs"),
            pytest.param(1, "lhs_scale", [16, 2], DataType.FP8E8M0, [16], id="lhs-scale"),
            pytest.param(2, "rhs", [64, 32], DataType.FP8E4M3FN, [64], id="rhs"),
            pytest.param(3, "rhs_scale", [2, 32], DataType.FP8E8M0, [2], id="rhs-scale"),
        ],
    )
    def test_rejects_rank_mismatched_operand_valid_shape(self, index, name, shape, dtype, valid_shape):
        span = ir.Span.unknown()
        operands = list(self._mx_operands(span))
        operands[index] = _tile(name, shape, dtype, valid_shape=valid_shape)

        with pytest.raises(ValueError, match=r"valid_shape rank \(1\) must match.*rank \(2\)"):
            ir.op.tile.matmul_mx(*operands, span)

    def test_rejects_rank_mismatched_acc_and_bias_valid_shape(self):
        span = ir.Span.unknown()
        operands = self._mx_operands(span)

        with pytest.raises(ValueError, match=r"valid_shape rank \(1\) must match.*rank \(2\)"):
            ir.op.tile.matmul_mx_acc(_tile("acc", [16, 32], DataType.FP32, valid_shape=[16]), *operands, span)

        with pytest.raises(ValueError, match=r"valid_shape rank \(1\) must match.*rank \(2\)"):
            ir.op.tile.matmul_mx_bias(*operands, _tile("bias", [1, 32], DataType.FP32, valid_shape=[1]), span)

    def test_rejects_bias_valid_mismatch(self):
        span = ir.Span.unknown()
        lhs, lhs_scale, rhs, rhs_scale = self._mx_operands(span, valid=(8, 64, 16))
        bias = _tile("bias", [1, 32], DataType.FP32, valid_shape=[1, 32])
        with pytest.raises(ValueError, match="bias valid cols"):
            ir.op.tile.matmul_mx_bias(lhs, lhs_scale, rhs, rhs_scale, bias, span)


# The four MX data pairs: PTOAS v0.72 `isA5MxFp8InputType` admits E4M3FN and E5M2
# per side independently, so every combination is legal and HF8 is not.
_MX_FP8_PAIRS = [
    pytest.param(DataType.FP8E4M3FN, DataType.FP8E4M3FN, id="e4m3-x-e4m3"),
    pytest.param(DataType.FP8E5M2, DataType.FP8E5M2, id="e5m2-x-e5m2"),
    pytest.param(DataType.FP8E4M3FN, DataType.FP8E5M2, id="e4m3-x-e5m2"),
    pytest.param(DataType.FP8E5M2, DataType.FP8E4M3FN, id="e5m2-x-e4m3"),
]


class TestGemvMxRegistry:
    """Memory spaces and A5-only registration for the MX GEMV family."""

    def test_memory_specs(self):
        base = ir.get_op_memory_spec("tile.gemv_mx")
        assert base is not None
        assert base["output_memory"] == ir.MemorySpace.Acc
        assert base["input_constraints"] == [
            [ir.MemorySpace.Left],
            [ir.MemorySpace.LeftScale],
            [ir.MemorySpace.Right],
            [ir.MemorySpace.RightScale],
        ]
        acc = ir.get_op_memory_spec("tile.gemv_mx_acc")
        assert acc is not None
        assert acc["input_constraints"][0] == [ir.MemorySpace.Acc]
        bias = ir.get_op_memory_spec("tile.gemv_mx_bias")
        assert bias is not None
        assert bias["input_constraints"][4] == [ir.MemorySpace.Bias]


class TestGemvMxTypes:
    """Type deduction for `tile.gemv_mx` / `_acc` / `_bias`.

    Physical K is 512 throughout, which is what the Mat-to-Left vector extract of
    a single-row lhs requires for a 1-byte operand (pto-isa
    `npu/a5/TExtractCommon.hpp:182-183`, `Cols % (512 / sizeof(DType)) == 0`), so
    the shapes here are the ones that can actually be assembled.
    """

    K = 512
    N = 96
    G = K // 32

    @classmethod
    def _operands(
        cls, span, *, k=None, n=None, m=1, lhs_dtype=DataType.FP8E4M3FN, rhs_dtype=None, valid=None
    ):
        """Build a valid MX GEMV operand set; optional valid=(k_v, n_v)."""
        k = cls.K if k is None else k
        n = cls.N if n is None else n
        rhs_dtype = lhs_dtype if rhs_dtype is None else rhs_dtype
        groups = (k + 31) // 32
        if valid is None:
            lhs = _tile("lhs", [m, k], lhs_dtype)
            lhs_scale = _tile("lhs_scale", [m, groups], DataType.FP8E8M0)
            rhs = _tile("rhs", [k, n], rhs_dtype)
            rhs_scale = _tile("rhs_scale", [groups, n], DataType.FP8E8M0)
            return lhs, lhs_scale, rhs, rhs_scale
        k_v, n_v = valid
        groups_v = (k_v + 31) // 32
        lhs = _tile("lhs", [m, k], lhs_dtype, valid_shape=[m, k_v])
        lhs_scale = _tile("lhs_scale", [m, groups], DataType.FP8E8M0, valid_shape=[m, groups_v])
        rhs = _tile("rhs", [k, n], rhs_dtype, valid_shape=[k_v, n_v])
        rhs_scale = _tile("rhs_scale", [groups, n], DataType.FP8E8M0, valid_shape=[groups_v, n_v])
        return lhs, lhs_scale, rhs, rhs_scale

    @pytest.mark.parametrize(("lhs_dtype", "rhs_dtype"), _MX_FP8_PAIRS)
    def test_every_fp8_pair_deduces_in_every_form(self, lhs_dtype, rhs_dtype):
        """All four pairs x base/acc/bias land in the 16-row Acc box with valid [1, N]."""
        span = ir.Span.unknown()
        lhs, lhs_scale, rhs, rhs_scale = self._operands(span, lhs_dtype=lhs_dtype, rhs_dtype=rhs_dtype)

        base = ir.op.tile.gemv_mx(lhs, lhs_scale, rhs, rhs_scale, span=span)
        base_type = base.type
        assert isinstance(base_type, ir.TileType)
        assert base_type.dtype == DataType.FP32
        rows, cols = base_type.shape
        # 16 physical rows: a GEMV result is one logical row in the mandatory
        # NZ-boxed L0C accumulator, and that physical/logical split is what the
        # codegen bridge re-types for PTOAS.
        assert isinstance(rows, ir.ConstInt) and rows.value == 16
        assert isinstance(cols, ir.ConstInt) and cols.value == self.N
        assert base_type.tile_view is not None
        valid_rows, valid_cols = base_type.tile_view.valid_shape
        assert isinstance(valid_rows, ir.ConstInt) and valid_rows.value == 1
        assert isinstance(valid_cols, ir.ConstInt) and valid_cols.value == self.N
        assert base_type.tile_view.fractal == 1024

        acc = _tile("acc", [16, self.N], DataType.FP32, valid_shape=[1, self.N])
        acc_call = ir.op.tile.gemv_mx_acc(acc, lhs, lhs_scale, rhs, rhs_scale, span=span)
        assert acc_call.op.name == ir.get_op("tile.gemv_mx_acc").name
        assert isinstance(acc_call.type, ir.TileType)

        bias = _tile("bias", [1, self.N], DataType.FP32)
        bias_call = ir.op.tile.gemv_mx_bias(lhs, lhs_scale, rhs, rhs_scale, bias, span=span)
        assert bias_call.op.name == ir.get_op("tile.gemv_mx_bias").name
        assert isinstance(bias_call.type, ir.TileType)

    def test_requires_exactly_one_physical_row(self):
        """The Cube reads GEMV mode off the Left tile's *physical* row count.

        `GetGemvCtrl` in pto-isa `a5/TMatmulCommon.hpp:30-33` is
        `TileLeft::Rows != 1`, so a 16-row lhs would run as a matmul even with one
        valid row -- which is why this is rejected rather than narrowed.
        """
        span = ir.Span.unknown()
        with pytest.raises(ValueError, match="requires physical M to be exactly 1"):
            ir.op.tile.gemv_mx(*self._operands(span, m=16), span=span)

    @pytest.mark.parametrize("rejected", [DataType.HF8, DataType.FP16, DataType.BF16, DataType.INT8])
    def test_rejects_non_mx_data_dtypes(self, rejected):
        """HF8 reaches ordinary `gemv` but not `gemv_mx` -- PTOAS gates MX separately."""
        span = ir.Span.unknown()
        _, lhs_scale, rhs, rhs_scale = self._operands(span)
        bad_lhs = _tile("bad_lhs", [1, self.K], rejected)
        with pytest.raises(ValueError, match="requires lhs dtype FP8E4M3FN or FP8E5M2"):
            ir.op.tile.gemv_mx(bad_lhs, lhs_scale, rhs, rhs_scale, span=span)

        lhs, lhs_scale, _, rhs_scale = self._operands(span)
        bad_rhs = _tile("bad_rhs", [self.K, self.N], rejected)
        with pytest.raises(ValueError, match="requires rhs dtype FP8E4M3FN or FP8E5M2"):
            ir.op.tile.gemv_mx(lhs, lhs_scale, bad_rhs, rhs_scale, span=span)

    @pytest.mark.parametrize("dtype", [DataType.FP4E2M1X2, DataType.FP4E1M2X2])
    def test_rejects_native_packed_fp4(self, dtype):
        span = ir.Span.unknown()
        with pytest.raises(ValueError, match=r"requires lhs dtype FP8E4M3FN or FP8E5M2.*packed FP4.*pl.cast"):
            ir.op.tile.gemv_mx(*self._operands(span, lhs_dtype=dtype), span=span)

    def test_rejects_reused_scale_between_operands(self):
        span = ir.Span.unknown()
        lhs, scale, rhs, _ = self._operands(span)
        with pytest.raises(ValueError, match="requires distinct lhs_scale and rhs_scale tiles"):
            ir.op.tile.gemv_mx(lhs, scale, rhs, scale, span=span)

    def test_rejects_misaligned_physical_k_and_n(self):
        span = ir.Span.unknown()
        with pytest.raises(ValueError, match="divisible by 64"):
            ir.op.tile.gemv_mx(*self._operands(span, k=96), span=span)
        with pytest.raises(ValueError, match=r"physical N divisible by 32"):
            ir.op.tile.gemv_mx(*self._operands(span, n=16), span=span)

    @pytest.mark.parametrize(
        ("scale_shape", "which", "error"),
        [
            pytest.param([16, 16], "lhs_scale", "lhs_scale physical rows", id="lhs-scale-16-rows"),
            pytest.param([1, 8], "lhs_scale", "lhs_scale physical cols", id="lhs-scale-wrong-groups"),
            pytest.param([8, 96], "rhs_scale", "rhs_scale physical rows", id="rhs-scale-wrong-groups"),
            pytest.param([16, 32], "rhs_scale", "rhs_scale physical cols", id="rhs-scale-wrong-n"),
        ],
    )
    def test_rejects_wrong_scale_geometry(self, scale_shape, which, error):
        """The A scale is [1, ceil(K/32)] and the B scale [ceil(K/32), N].

        A 16-row A scale is the shape someone reaches for by padding an `MX_A_ZZ`
        buffer to its block granularity; PTOAS rejects it
        (`expects a_scale shape/valid_shape to be [M, ceil(K/32)]`), so PyPTO does
        too rather than emitting it.
        """
        span = ir.Span.unknown()
        lhs, lhs_scale, rhs, rhs_scale = self._operands(span)
        replacement = _tile(which, scale_shape, DataType.FP8E8M0)
        if which == "lhs_scale":
            lhs_scale = replacement
        else:
            rhs_scale = replacement
        with pytest.raises(ValueError, match=error):
            ir.op.tile.gemv_mx(lhs, lhs_scale, rhs, rhs_scale, span=span)

    def test_rejects_non_fp8e8m0_scale(self):
        span = ir.Span.unknown()
        lhs, _, rhs, rhs_scale = self._operands(span)
        with pytest.raises(ValueError, match="FP8E8M0"):
            ir.op.tile.gemv_mx(lhs, _tile("ls", [1, self.G], DataType.FP16), rhs, rhs_scale, span=span)

    @pytest.mark.parametrize(("lhs_dtype", "rhs_dtype"), _MX_FP8_PAIRS)
    def test_rejects_every_narrowed_valid_k(self, lhs_dtype, rhs_dtype):
        """Valid K must EQUAL physical K for the MX GEMV family, per pair.

        This assertion was inverted on purpose. It previously accepted
        `valid K = 481` against `physical K = 512` on the grounds that both round
        up to 16 scale groups, which is what `gemv_mx` and `tget_scale_addr` need
        to agree on. Agreement is necessary but not sufficient: a narrowed valid K
        assembles and then faults the A5 card, for every FP8 pair
        (docs/en/dev/ir/05-operators.md, "On device, valid K must equal physical
        K"). A test that admits a card-faulting shape is worse than no test, so the
        shape it used to accept is now the shape it requires to be rejected.

        Both the group-count-preserving narrowing (481 -> 16 groups, same as 512)
        and the group-count-changing one (480 -> 15 groups) are rejected; only the
        messages differ, since the second trips the older scale-group check first.
        """
        span = ir.Span.unknown()
        same_group_count = self._operands(
            span, lhs_dtype=lhs_dtype, rhs_dtype=rhs_dtype, valid=(self.K - 31, self.N)
        )
        with pytest.raises(ValueError, match="valid K to equal physical K"):
            ir.op.tile.gemv_mx(*same_group_count, span=span)

        fewer_groups = self._operands(
            span, lhs_dtype=lhs_dtype, rhs_dtype=rhs_dtype, valid=(self.K - 32, self.N)
        )
        with pytest.raises(ValueError, match="same scale-group count as physical K"):
            ir.op.tile.gemv_mx(*fewer_groups, span=span)

        # Full valid K stays accepted -- the rejection is about narrowing K, not
        # about carrying a valid_shape at all (a narrowed valid N is still fine).
        full = self._operands(span, lhs_dtype=lhs_dtype, rhs_dtype=rhs_dtype, valid=(self.K, self.N))
        assert isinstance(ir.op.tile.gemv_mx(*full, span=span).type, ir.TileType)

    def test_narrowed_valid_k_is_rejected_in_the_acc_and_bias_forms_too(self):
        """All three GEMV forms share the deducer, so all three must reject it."""
        span = ir.Span.unknown()
        lhs, lhs_scale, rhs, rhs_scale = self._operands(span, valid=(self.K - 31, self.N))
        acc = _tile("acc", [16, self.N], DataType.FP32, valid_shape=[1, self.N])
        bias = _tile("bias", [1, self.N], DataType.FP32)
        with pytest.raises(ValueError, match="valid K to equal physical K"):
            ir.op.tile.gemv_mx_acc(acc, lhs, lhs_scale, rhs, rhs_scale, span=span)
        with pytest.raises(ValueError, match="valid K to equal physical K"):
            ir.op.tile.gemv_mx_bias(lhs, lhs_scale, rhs, rhs_scale, bias, span=span)

    def test_matmul_mx_still_accepts_a_narrowed_valid_k(self):
        """The rejection above is scoped to the GEMV family, and that is deliberate.

        `tile.matmul_mx` has carried the looser contract since before this family
        existed, and pypto-lib's `deepseek_v4_1_flash` MX models are built on it.
        Tightening matmul is a contract change for existing callers, not a fix to
        this PR, so it is not made here -- this test pins that scope so the next
        person sees the split is intentional rather than an oversight.
        """
        span = ir.Span.unknown()
        narrowed = ir.op.tile.matmul_mx(*TestMatmulMxTypes._mx_operands(span, valid=(16, 48, 32)), span)
        assert isinstance(narrowed.type, ir.TileType)

    def test_rejects_acc_phase_on_the_bias_form_through_a_raw_call(self):
        """`pto.tgemv.mx.bias` has no accPhase attribute, so the kwarg must not pass.

        The Python wrapper `pl.tile.gemv_mx_bias` has no `acc_phase` parameter, but
        the op registration declares no attribute schema, so a raw
        `create_op_call` carrying the keyword would otherwise reach deduction and
        be silently dropped on the way to PTOAS.
        """
        span = ir.Span.unknown()
        lhs, lhs_scale, rhs, rhs_scale = self._operands(span)
        bias = _tile("bias", [1, self.N], DataType.FP32)
        with pytest.raises(ValueError, match="does not accept acc_phase"):
            ir.create_op_call(
                "tile.gemv_mx_bias",
                [lhs, lhs_scale, rhs, rhs_scale, bias],
                {"acc_phase": int(ir.AccPhase.Partial)},
                span,
            )

    def test_rejects_acc_shape_and_dtype_mismatch(self):
        span = ir.Span.unknown()
        lhs, lhs_scale, rhs, rhs_scale = self._operands(span)
        with pytest.raises(ValueError, match="requires acc dtype FP32"):
            ir.op.tile.gemv_mx_acc(
                _tile("acc", [16, self.N], DataType.FP16, valid_shape=[1, self.N]),
                lhs,
                lhs_scale,
                rhs,
                rhs_scale,
                span=span,
            )
        with pytest.raises(ValueError, match="acc cols to match output N"):
            ir.op.tile.gemv_mx_acc(
                _tile("acc", [16, 32], DataType.FP32, valid_shape=[1, 32]),
                lhs,
                lhs_scale,
                rhs,
                rhs_scale,
                span=span,
            )
        with pytest.raises(ValueError, match="acc valid rows"):
            ir.op.tile.gemv_mx_acc(
                _tile("acc", [16, self.N], DataType.FP32, valid_shape=[16, self.N]),
                lhs,
                lhs_scale,
                rhs,
                rhs_scale,
                span=span,
            )

    def test_rejects_bias_shape_and_dtype_mismatch(self):
        span = ir.Span.unknown()
        lhs, lhs_scale, rhs, rhs_scale = self._operands(span)
        with pytest.raises(ValueError, match="requires bias dtype FP32"):
            ir.op.tile.gemv_mx_bias(
                lhs, lhs_scale, rhs, rhs_scale, _tile("bias", [1, self.N], DataType.FP16), span=span
            )
        with pytest.raises(ValueError, match=r"requires bias shape \[1, N\]"):
            ir.op.tile.gemv_mx_bias(
                lhs, lhs_scale, rhs, rhs_scale, _tile("bias", [16, self.N], DataType.FP32), span=span
            )
        with pytest.raises(ValueError, match="bias cols to match N"):
            ir.op.tile.gemv_mx_bias(
                lhs, lhs_scale, rhs, rhs_scale, _tile("bias", [1, 32], DataType.FP32), span=span
            )

    # AccPhase's integer values are the PTO-ISA ones: 0 unspecified, 2 partial,
    # 3 final. 1 is the gap between them, which is why it is in this list.
    @pytest.mark.parametrize("bad_phase", [-1, 1, 4, 99])
    def test_rejects_invalid_acc_phase(self, bad_phase):
        """`gemv_mx` and `_acc` carry acc_phase; `_bias` has no such parameter.

        `pto.tgemv.mx.bias` declares no accPhase attribute, so an emitted one
        would be kept as a discardable MLIR attribute and silently ignored --
        the phase is therefore not accepted at this layer at all.
        """
        span = ir.Span.unknown()
        lhs, lhs_scale, rhs, rhs_scale = self._operands(span)
        with pytest.raises(ValueError, match="requires acc_phase to be"):
            ir.op.tile.gemv_mx(lhs, lhs_scale, rhs, rhs_scale, acc_phase=bad_phase, span=span)

        acc = _tile("acc", [16, self.N], DataType.FP32, valid_shape=[1, self.N])
        with pytest.raises(ValueError, match="requires acc_phase to be"):
            ir.op.tile.gemv_mx_acc(acc, lhs, lhs_scale, rhs, rhs_scale, acc_phase=bad_phase, span=span)

    def test_bias_form_has_no_acc_phase_parameter(self):
        span = ir.Span.unknown()
        lhs, lhs_scale, rhs, rhs_scale = self._operands(span)
        bias = _tile("bias", [1, self.N], DataType.FP32)
        with pytest.raises(TypeError, match="acc_phase"):
            ir.op.tile.gemv_mx_bias(lhs, lhs_scale, rhs, rhs_scale, bias, acc_phase=0, span=span)

    def test_single_row_scale_load_takes_the_mx_scale_box(self):
        """A plain-ND one-row FP8E8M0 Mat load is an MX A scale, not a data vector.

        A single-row Mat *data* load gets row_major / none_box (the cube vector
        path). An FP8E8M0 one does not: it has to reach LeftScale, which accepts
        only the row/row/32 scale box. `MX_A_ZZ` cannot supply it because that
        layout blocks 16 rows x 2 groups per 32-byte fractal, which has no meaning
        for one row -- and the hardware reads a one-row scale as ceil(K/32)
        *contiguous* bytes, so padding a packed buffer to 16 rows would read the
        wrong bytes.
        """
        span = ir.Span.unknown()
        scale_tensor = ir.Var("a_scale", ir.TensorType([1, self.G], DataType.FP8E8M0), span)
        scale_load = ir.op.tile.load(
            scale_tensor, [0, 0], [1, self.G], target_memory=ir.MemorySpace.Mat, span=span
        )
        scale_type = scale_load.type
        assert isinstance(scale_type, ir.TileType)
        assert scale_type.tile_view is not None
        assert scale_type.tile_view.fractal == 32
        assert scale_type.tile_view.blayout == ir.TileLayout.row_major
        assert scale_type.tile_view.slayout == ir.TileLayout.row_major

        # The data vector beside it keeps the ND row-vector view it always had.
        data_tensor = ir.Var("a", ir.TensorType([1, self.K], DataType.FP8E4M3FN), span)
        data_load = ir.op.tile.load(
            data_tensor, [0, 0], [1, self.K], target_memory=ir.MemorySpace.Mat, span=span
        )
        data_type = data_load.type
        assert isinstance(data_type, ir.TileType)
        assert data_type.tile_view is not None
        assert data_type.tile_view.slayout == ir.TileLayout.none_box

        # And a multi-row FP8E8M0 plain-ND load is left alone: there is no
        # unpacked form for it, so it must not be mistaken for a scale box.
        wide_tensor = ir.Var("wide", ir.TensorType([16, self.G], DataType.FP8E8M0), span)
        wide_load = ir.op.tile.load(
            wide_tensor, [0, 0], [16, self.G], target_memory=ir.MemorySpace.Mat, span=span
        )
        wide_type = wide_load.type
        assert isinstance(wide_type, ir.TileType)
        assert wide_type.tile_view is None or wide_type.tile_view.fractal != 32


class TestTGetScaleAddr:
    def test_binds_and_rejects(self):
        span = ir.Span.unknown()
        dst = _tile("rbs", [2, 32], DataType.FP8E8M0, memory=ir.MemorySpace.RightScale)
        src = _tile("rb", [64, 32], DataType.FP8E4M3FN, memory=ir.MemorySpace.Right)
        call = ir.op.tile.tget_scale_addr(dst, src, span)
        call_type = call.type
        assert isinstance(call_type, ir.TileType)
        assert call_type.memory_space == ir.MemorySpace.RightScale

        fp4_src = _tile("fp4_rb", [64, 32], DataType.FP4E2M1X2, memory=ir.MemorySpace.Right)
        with pytest.raises(ValueError, match="requires src dtype FP8E4M3FN"):
            ir.op.tile.tget_scale_addr(dst, fp4_src, span)

        with pytest.raises(ValueError, match="LeftScale↔Left|RightScale↔Right"):
            ir.op.tile.tget_scale_addr(
                _tile("las", [16, 2], DataType.FP8E8M0, memory=ir.MemorySpace.LeftScale),
                _tile("rb", [64, 32], DataType.FP8E4M3FN, memory=ir.MemorySpace.Right),
                span,
            )
        with pytest.raises(ValueError, match="FP8E8M0"):
            ir.op.tile.tget_scale_addr(
                _tile("las", [16, 2], DataType.UINT8, memory=ir.MemorySpace.LeftScale),
                _tile("la", [16, 64], DataType.FP8E4M3FN, memory=ir.MemorySpace.Left),
                span,
            )
        with pytest.raises(ValueError, match="requires src dtype FP8E4M3FN"):
            ir.op.tile.tget_scale_addr(
                _tile("las", [16, 2], DataType.FP8E8M0, memory=ir.MemorySpace.LeftScale),
                _tile("la", [16, 64], DataType.FP16, memory=ir.MemorySpace.Left),
                span,
            )
        with pytest.raises(ValueError, match="physical cols|dst_scale shape"):
            ir.op.tile.tget_scale_addr(
                _tile("las", [16, 1], DataType.FP8E8M0, memory=ir.MemorySpace.LeftScale),
                _tile("la", [16, 64], DataType.FP8E4M3FN, memory=ir.MemorySpace.Left),
                span,
            )

    def test_requires_resolved_memory_spaces(self):
        span = ir.Span.unknown()
        with pytest.raises(ValueError, match="resolved dst_scale and src memory spaces"):
            ir.op.tile.tget_scale_addr(
                _tile("scale", [16, 2], DataType.FP8E8M0),
                _tile("data", [16, 64], DataType.FP8E4M3FN),
                span,
            )

    @pytest.mark.parametrize("malformed_operand", ["scale", "data"])
    def test_rejects_rank_mismatched_valid_shape(self, malformed_operand):
        span = ir.Span.unknown()
        dst = _tile(
            "las",
            [16, 2],
            DataType.FP8E8M0,
            valid_shape=[16] if malformed_operand == "scale" else [16, 2],
            memory=ir.MemorySpace.LeftScale,
        )
        src = _tile(
            "la",
            [16, 64],
            DataType.FP8E4M3FN,
            valid_shape=[16] if malformed_operand == "data" else [16, 64],
            memory=ir.MemorySpace.Left,
        )

        with pytest.raises(ValueError, match=r"valid_shape rank \(1\) must match.*rank \(2\)"):
            ir.op.tile.tget_scale_addr(dst, src, span)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
