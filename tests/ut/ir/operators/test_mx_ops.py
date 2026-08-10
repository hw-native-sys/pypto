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


class TestMxDtypeAndMemorySpace:
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


def _tile(name, shape, dtype, *, valid_shape=None, memory=None):
    span = ir.Span.unknown()
    view = None
    if valid_shape is not None:
        view = ir.TileView(valid_shape=valid_shape, stride=[])
    return ir.Var(name, ir.TileType(shape, dtype, tile_view=view, memory_space=memory), span)


def _mx_operands(span, *, m=16, k=64, n=32, valid=None):
    """Build a valid MX operand set; optional valid=(m_v, k_v, n_v)."""
    if valid is None:
        lhs = ir.Var("lhs", ir.TileType([m, k], DataType.FP8E4M3FN), span)
        lhs_scale = ir.Var("lhs_scale", ir.TileType([m, k // 32], DataType.FP8E8M0), span)
        rhs = ir.Var("rhs", ir.TileType([k, n], DataType.FP8E4M3FN), span)
        rhs_scale = ir.Var("rhs_scale", ir.TileType([k // 32, n], DataType.FP8E8M0), span)
        return lhs, lhs_scale, rhs, rhs_scale
    m_v, k_v, n_v = valid
    sk = (k + 31) // 32
    sk_v = (k_v + 31) // 32
    lhs = _tile("lhs", [m, k], DataType.FP8E4M3FN, valid_shape=[m_v, k_v])
    lhs_scale = _tile("lhs_scale", [m, sk], DataType.FP8E8M0, valid_shape=[m_v, sk_v])
    rhs = _tile("rhs", [k, n], DataType.FP8E4M3FN, valid_shape=[k_v, n_v])
    rhs_scale = _tile("rhs_scale", [sk, n], DataType.FP8E8M0, valid_shape=[sk_v, n_v])
    return lhs, lhs_scale, rhs, rhs_scale


class TestMatmulMxRegistry:
    def test_memory_specs(self):
        assert not hasattr(pl, "tget_scale_addr")
        mx = ir.get_op_memory_spec("tile.matmul_mx")
        assert mx["output_memory"] == ir.MemorySpace.Acc
        assert mx["input_constraints"] == [
            [ir.MemorySpace.Left],
            [ir.MemorySpace.LeftScale],
            [ir.MemorySpace.Right],
            [ir.MemorySpace.RightScale],
        ]
        acc = ir.get_op_memory_spec("tile.matmul_mx_acc")
        assert acc["input_constraints"][0] == [ir.MemorySpace.Acc]
        bias = ir.get_op_memory_spec("tile.matmul_mx_bias")
        assert bias["input_constraints"][4] == [ir.MemorySpace.Bias]
        tget = ir.get_op_memory_spec("tile.tget_scale_addr")
        assert tget["input_constraints"] == []
        assert tget["output_memory"] == "inherit_from_input"


class TestMatmulMxTypes:
    def test_type_deduction_and_variants(self):
        span = ir.Span.unknown()
        lhs, lhs_scale, rhs, rhs_scale = _mx_operands(span)
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

    def test_accepts_packed_flat_scale_shapes(self):
        """quant_mx(layout) packed-flat [1,G] scales are accepted by matmul_mx*."""
        span = ir.Span.unknown()
        m, k, n = 16, 64, 32
        lhs = ir.Var("lhs", ir.TileType([m, k], DataType.FP8E4M3FN), span)
        lhs_scale = ir.Var("lhs_scale", ir.TileType([1, m * (k // 32)], DataType.FP8E8M0), span)
        rhs = ir.Var("rhs", ir.TileType([k, n], DataType.FP8E4M3FN), span)
        rhs_scale = ir.Var("rhs_scale", ir.TileType([1, (k // 32) * n], DataType.FP8E8M0), span)
        call = ir.op.tile.matmul_mx(lhs, lhs_scale, rhs, rhs_scale, span)
        assert isinstance(call.type, ir.TileType)
        assert call.type.dtype == DataType.FP32

    def test_rejects_wrong_dtypes_and_alignment(self):
        span = ir.Span.unknown()
        # data must be FP8E4M3FN
        with pytest.raises(ValueError, match="FP8E4M3FN"):
            ir.op.tile.matmul_mx(
                ir.Var("lhs", ir.TileType([16, 64], DataType.FP8E5M2), span),
                ir.Var("ls", ir.TileType([16, 2], DataType.FP8E8M0), span),
                ir.Var("rhs", ir.TileType([64, 32], DataType.FP8E5M2), span),
                ir.Var("rs", ir.TileType([2, 32], DataType.FP8E8M0), span),
                span,
            )
        # scale must be FP8E8M0
        lhs, _, rhs, rhs_scale = _mx_operands(span)
        with pytest.raises(ValueError, match="FP8E8M0"):
            ir.op.tile.matmul_mx(
                lhs, ir.Var("ls", ir.TileType([16, 2], DataType.FP16), span), rhs, rhs_scale, span
            )
        # M%16 / K%64 / N%32
        with pytest.raises(ValueError, match="divisible by 16"):
            ir.op.tile.matmul_mx(*_mx_operands(span, m=8), span)
        with pytest.raises(ValueError, match="divisible by 64"):
            ir.op.tile.matmul_mx(*_mx_operands(span, k=96), span)
        with pytest.raises(ValueError, match="divisible by 32"):
            ir.op.tile.matmul_mx(*_mx_operands(span, n=16), span)

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
        call = ir.op.tile.matmul_mx(*_mx_operands(span, valid=(16, 48, 32)), span)
        call_type = call.type
        assert isinstance(call_type, ir.TileType)
        assert call_type.dtype == DataType.FP32

        # propagate contracted M/N valid into Acc output
        call = ir.op.tile.matmul_mx(*_mx_operands(span, valid=(8, 64, 16)), span)
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
            lhs, _, rhs, rhs_scale = _mx_operands(span)
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
            ir.op.tile.matmul_mx(*_mx_operands(span, valid=(16, 0, 32)), span)

        # ceil(validK/32) must equal ceil(physicalK/32) (PTOAS matmul_mx vs tget)
        with pytest.raises(ValueError, match="scale-group count"):
            ir.op.tile.matmul_mx(*_mx_operands(span, valid=(16, 31, 32)), span)

        # acc valid_shape must match MX output valid M/N
        acc = _tile("acc", [16, 32], DataType.FP32, valid_shape=[16, 32])
        with pytest.raises(ValueError, match="acc valid rows"):
            ir.op.tile.matmul_mx_acc(acc, *_mx_operands(span, valid=(8, 64, 16)), span)

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
        operands = list(_mx_operands(span))
        operands[index] = _tile(name, shape, dtype, valid_shape=valid_shape)

        with pytest.raises(ValueError, match=r"valid_shape rank \(1\) must match.*rank \(2\)"):
            ir.op.tile.matmul_mx(*operands, span)

    def test_rejects_rank_mismatched_acc_and_bias_valid_shape(self):
        span = ir.Span.unknown()
        operands = _mx_operands(span)

        with pytest.raises(ValueError, match=r"valid_shape rank \(1\) must match.*rank \(2\)"):
            ir.op.tile.matmul_mx_acc(_tile("acc", [16, 32], DataType.FP32, valid_shape=[16]), *operands, span)

        with pytest.raises(ValueError, match=r"valid_shape rank \(1\) must match.*rank \(2\)"):
            ir.op.tile.matmul_mx_bias(*operands, _tile("bias", [1, 32], DataType.FP32, valid_shape=[1]), span)

    def test_rejects_bias_valid_mismatch(self):
        span = ir.Span.unknown()
        lhs, lhs_scale, rhs, rhs_scale = _mx_operands(span, valid=(8, 64, 16))
        bias = _tile("bias", [1, 32], DataType.FP32, valid_shape=[1, 32])
        with pytest.raises(ValueError, match="bias valid cols"):
            ir.op.tile.matmul_mx_bias(lhs, lhs_scale, rhs, rhs_scale, bias, span)


class TestTGetScaleAddr:
    def test_binds_and_rejects(self):
        span = ir.Span.unknown()
        dst = _tile("rbs", [2, 32], DataType.FP8E8M0, memory=ir.MemorySpace.RightScale)
        src = _tile("rb", [64, 32], DataType.FP8E4M3FN, memory=ir.MemorySpace.Right)
        call = ir.op.tile.tget_scale_addr(dst, src, span)
        call_type = call.type
        assert isinstance(call_type, ir.TileType)
        assert call_type.memory_space == ir.MemorySpace.RightScale

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
        with pytest.raises(ValueError, match="FP8E4M3FN"):
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
