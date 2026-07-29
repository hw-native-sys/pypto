# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for minimal MXFP8 matmul path: matmul_mx, tget_scale_addr, mx load."""

import pypto.language as pl
import pytest
from pypto import ir
from pypto.pypto_core import DataType


def _tile(name, shape, dtype, *, valid_shape=None, memory=None):
    span = ir.Span.unknown()
    view = None
    if valid_shape is not None:
        view = ir.TileView(valid_shape=valid_shape, stride=[], start_offset=None)
    return ir.Var(name, ir.TileType(shape, dtype, tile_view=view, memory_space=memory), span)


class TestMatmulMxRegistry:
    def test_matmul_mx_spec(self):
        spec = ir.get_op_memory_spec("tile.matmul_mx")
        assert spec is not None
        assert spec["output_memory"] == ir.MemorySpace.Acc
        constraints = spec["input_constraints"]
        assert constraints[0] == [ir.MemorySpace.Left]
        assert constraints[1] == [ir.MemorySpace.LeftScale]
        assert constraints[2] == [ir.MemorySpace.Right]
        assert constraints[3] == [ir.MemorySpace.RightScale]


class TestMatmulMxTypes:
    def test_matmul_mx_type_deduction(self):
        span = ir.Span.unknown()
        lhs = ir.Var("lhs", ir.TileType([16, 64], DataType.FP8E4M3FN), span)
        lhs_scale = ir.Var("lhs_scale", ir.TileType([16, 2], DataType.FP8E8M0), span)
        rhs = ir.Var("rhs", ir.TileType([64, 32], DataType.FP8E4M3FN), span)
        rhs_scale = ir.Var("rhs_scale", ir.TileType([2, 32], DataType.FP8E8M0), span)
        call = ir.op.tile.matmul_mx(lhs, lhs_scale, rhs, rhs_scale, span)
        assert isinstance(call.type, ir.TileType)
        assert call.type.dtype == DataType.FP32
        assert isinstance(call.type.shape[0], ir.ConstInt) and call.type.shape[0].value == 16
        assert isinstance(call.type.shape[1], ir.ConstInt) and call.type.shape[1].value == 32

    def test_matmul_mx_rejects_fp8e5m2_data(self):
        span = ir.Span.unknown()
        lhs = ir.Var("lhs", ir.TileType([16, 64], DataType.FP8E5M2), span)
        lhs_scale = ir.Var("lhs_scale", ir.TileType([16, 2], DataType.FP8E8M0), span)
        rhs = ir.Var("rhs", ir.TileType([64, 32], DataType.FP8E5M2), span)
        rhs_scale = ir.Var("rhs_scale", ir.TileType([2, 32], DataType.FP8E8M0), span)
        with pytest.raises(Exception, match="FP8E4M3FN"):
            ir.op.tile.matmul_mx(lhs, lhs_scale, rhs, rhs_scale, span)

    def test_matmul_mx_rejects_bad_scale_dtype(self):
        span = ir.Span.unknown()
        lhs = ir.Var("lhs", ir.TileType([16, 64], DataType.FP8E4M3FN), span)
        lhs_scale = ir.Var("lhs_scale", ir.TileType([16, 2], DataType.FP16), span)
        rhs = ir.Var("rhs", ir.TileType([64, 32], DataType.FP8E4M3FN), span)
        rhs_scale = ir.Var("rhs_scale", ir.TileType([2, 32], DataType.FP8E8M0), span)
        with pytest.raises(Exception, match="FP8E8M0"):
            ir.op.tile.matmul_mx(lhs, lhs_scale, rhs, rhs_scale, span)

    def test_matmul_mx_rejects_physical_k_not_divisible_by_64(self):
        span = ir.Span.unknown()
        # physical K=96 is divisible by 32 but not by 64 — PTOAS rejects.
        lhs = ir.Var("lhs", ir.TileType([16, 96], DataType.FP8E4M3FN), span)
        lhs_scale = ir.Var("lhs_scale", ir.TileType([16, 3], DataType.FP8E8M0), span)
        rhs = ir.Var("rhs", ir.TileType([96, 32], DataType.FP8E4M3FN), span)
        rhs_scale = ir.Var("rhs_scale", ir.TileType([3, 32], DataType.FP8E8M0), span)
        with pytest.raises(Exception, match="divisible by 64"):
            ir.op.tile.matmul_mx(lhs, lhs_scale, rhs, rhs_scale, span)

    def test_matmul_mx_rejects_physical_m_not_divisible_by_16(self):
        span = ir.Span.unknown()
        lhs = ir.Var("lhs", ir.TileType([8, 64], DataType.FP8E4M3FN), span)
        lhs_scale = ir.Var("lhs_scale", ir.TileType([8, 2], DataType.FP8E8M0), span)
        rhs = ir.Var("rhs", ir.TileType([64, 32], DataType.FP8E4M3FN), span)
        rhs_scale = ir.Var("rhs_scale", ir.TileType([2, 32], DataType.FP8E8M0), span)
        with pytest.raises(Exception, match="divisible by 16"):
            ir.op.tile.matmul_mx(lhs, lhs_scale, rhs, rhs_scale, span)

    def test_matmul_mx_rejects_physical_n_not_divisible_by_32(self):
        span = ir.Span.unknown()
        lhs = ir.Var("lhs", ir.TileType([16, 64], DataType.FP8E4M3FN), span)
        lhs_scale = ir.Var("lhs_scale", ir.TileType([16, 2], DataType.FP8E8M0), span)
        rhs = ir.Var("rhs", ir.TileType([64, 16], DataType.FP8E4M3FN), span)
        rhs_scale = ir.Var("rhs_scale", ir.TileType([2, 16], DataType.FP8E8M0), span)
        with pytest.raises(Exception, match="divisible by 32"):
            ir.op.tile.matmul_mx(lhs, lhs_scale, rhs, rhs_scale, span)

    def test_matmul_mx_accepts_valid_k_not_multiple_of_32(self):
        # physical K=64, valid K=48 → scale groups ceil(48/32)=2.
        span = ir.Span.unknown()
        lhs = _tile("lhs", [16, 64], DataType.FP8E4M3FN, valid_shape=[16, 48])
        lhs_scale = _tile("lhs_scale", [16, 2], DataType.FP8E8M0, valid_shape=[16, 2])
        rhs = _tile("rhs", [64, 32], DataType.FP8E4M3FN, valid_shape=[48, 32])
        rhs_scale = _tile("rhs_scale", [2, 32], DataType.FP8E8M0, valid_shape=[2, 32])
        call = ir.op.tile.matmul_mx(lhs, lhs_scale, rhs, rhs_scale, span)
        assert isinstance(call.type, ir.TileType)
        assert call.type.dtype == DataType.FP32

    def test_matmul_mx_rejects_mismatched_valid_k(self):
        span = ir.Span.unknown()
        lhs = _tile("lhs", [16, 64], DataType.FP8E4M3FN, valid_shape=[16, 48])
        lhs_scale = _tile("lhs_scale", [16, 2], DataType.FP8E8M0)
        rhs = _tile("rhs", [64, 32], DataType.FP8E4M3FN, valid_shape=[40, 32])
        rhs_scale = _tile("rhs_scale", [2, 32], DataType.FP8E8M0)
        with pytest.raises(Exception, match="matching valid K"):
            ir.op.tile.matmul_mx(lhs, lhs_scale, rhs, rhs_scale, span)

    def test_matmul_mx_rejects_bad_scale_physical_shape(self):
        span = ir.Span.unknown()
        lhs = ir.Var("lhs", ir.TileType([16, 64], DataType.FP8E4M3FN), span)
        # physical cols should be ceil(64/32)=2, not 1
        lhs_scale = ir.Var("lhs_scale", ir.TileType([16, 1], DataType.FP8E8M0), span)
        rhs = ir.Var("rhs", ir.TileType([64, 32], DataType.FP8E4M3FN), span)
        rhs_scale = ir.Var("rhs_scale", ir.TileType([2, 32], DataType.FP8E8M0), span)
        with pytest.raises(Exception, match="physical cols"):
            ir.op.tile.matmul_mx(lhs, lhs_scale, rhs, rhs_scale, span)

    def test_matmul_mx_rejects_bad_scale_valid_shape(self):
        span = ir.Span.unknown()
        lhs = _tile("lhs", [16, 64], DataType.FP8E4M3FN, valid_shape=[16, 48])
        # physical OK [16,2], but valid cols should be ceil(48/32)=2, not 1
        lhs_scale = _tile("lhs_scale", [16, 2], DataType.FP8E8M0, valid_shape=[16, 1])
        rhs = _tile("rhs", [64, 32], DataType.FP8E4M3FN, valid_shape=[48, 32])
        rhs_scale = _tile("rhs_scale", [2, 32], DataType.FP8E8M0, valid_shape=[2, 32])
        with pytest.raises(Exception, match="valid cols"):
            ir.op.tile.matmul_mx(lhs, lhs_scale, rhs, rhs_scale, span)

    def test_matmul_mx_output_propagates_valid_shape(self):
        # M valid (8) < physical M (16), N valid (16) < physical N (32); scales aligned.
        # Output valid_shape must be the contracted {m_valid, n_valid} = {8, 16}.
        span = ir.Span.unknown()
        lhs = _tile("lhs", [16, 64], DataType.FP8E4M3FN, valid_shape=[8, 64])
        lhs_scale = _tile("lhs_scale", [16, 2], DataType.FP8E8M0, valid_shape=[8, 2])
        rhs = _tile("rhs", [64, 32], DataType.FP8E4M3FN, valid_shape=[64, 16])
        rhs_scale = _tile("rhs_scale", [2, 32], DataType.FP8E8M0, valid_shape=[2, 16])
        call = ir.op.tile.matmul_mx(lhs, lhs_scale, rhs, rhs_scale, span)
        assert isinstance(call.type, ir.TileType)
        vs = call.type.get_effective_tile_view().valid_shape
        assert len(vs) == 2
        assert isinstance(vs[0], ir.ConstInt) and vs[0].value == 8
        assert isinstance(vs[1], ir.ConstInt) and vs[1].value == 16

    def test_matmul_mx_rejects_nonpositive_valid_k(self):
        span = ir.Span.unknown()
        lhs = _tile("lhs", [16, 64], DataType.FP8E4M3FN, valid_shape=[16, 0])
        lhs_scale = _tile("lhs_scale", [16, 2], DataType.FP8E8M0)
        rhs = _tile("rhs", [64, 32], DataType.FP8E4M3FN, valid_shape=[0, 32])
        rhs_scale = _tile("rhs_scale", [2, 32], DataType.FP8E8M0)
        with pytest.raises(Exception, match="positive valid K"):
            ir.op.tile.matmul_mx(lhs, lhs_scale, rhs, rhs_scale, span)

    def test_matmul_mx_rejects_misaligned_valid_k(self):
        # physical K=64 (2 scale groups), valid K=31 (1 group): ceil(validK/32) !=
        # ceil(physicalK/32), so PTOAS matmul_mx and tget_scale_addr verifiers
        # conflict — reject at PyPTO rather than emit an unsatisfiable scale tile.
        span = ir.Span.unknown()
        lhs = _tile("lhs", [16, 64], DataType.FP8E4M3FN, valid_shape=[16, 31])
        lhs_scale = _tile("lhs_scale", [16, 2], DataType.FP8E8M0, valid_shape=[16, 2])
        rhs = _tile("rhs", [64, 32], DataType.FP8E4M3FN, valid_shape=[31, 32])
        rhs_scale = _tile("rhs_scale", [2, 32], DataType.FP8E8M0, valid_shape=[2, 32])
        with pytest.raises(Exception, match="scale-group count"):
            ir.op.tile.matmul_mx(lhs, lhs_scale, rhs, rhs_scale, span)


class TestTGetScaleAddr:
    def test_registered(self):
        spec = ir.get_op_memory_spec("tile.tget_scale_addr")
        assert spec is not None

    def test_inherits_rightscale(self):
        span = ir.Span.unknown()
        dst = _tile(
            "rbs",
            [2, 32],
            DataType.FP8E8M0,
            memory=ir.MemorySpace.RightScale,
        )
        src = _tile("rb", [64, 32], DataType.FP8E4M3FN, memory=ir.MemorySpace.Right)
        call = ir.op.tile.tget_scale_addr(dst, src, span)
        assert isinstance(call.type, ir.TileType)
        assert call.type.memory_space == ir.MemorySpace.RightScale

    def test_rejects_mismatched_pair(self):
        span = ir.Span.unknown()
        dst = _tile(
            "las",
            [16, 2],
            DataType.FP8E8M0,
            memory=ir.MemorySpace.LeftScale,
        )
        src = _tile("rb", [64, 32], DataType.FP8E4M3FN, memory=ir.MemorySpace.Right)
        with pytest.raises(Exception, match="LeftScale↔Left|pairing"):
            ir.op.tile.tget_scale_addr(dst, src, span)


class TestMxLoad:
    def test_mx_layout_sets_fractal(self):
        span = ir.Span.unknown()
        tensor = ir.Var("t", ir.TensorType([16, 2], DataType.FP8E8M0), span)
        call = ir.op.tile.load(
            tensor,
            [0, 0],
            [16, 2],
            target_memory=ir.MemorySpace.Mat,
            mx_layout="mx_a_zz",
            span=span,
        )
        assert isinstance(call.type, ir.TileType)
        assert call.type.dtype == DataType.FP8E8M0
        assert call.type.tile_view is not None
        assert call.type.tile_view.fractal == 32

    def test_rejects_unsupported_mx_layout(self):
        span = ir.Span.unknown()
        tensor = ir.Var("t", ir.TensorType([16, 2], DataType.FP8E8M0), span)
        with pytest.raises(Exception, match="mx_a_zz|mx_b_nn|mx_layout"):
            ir.op.tile.load(
                tensor,
                [0, 0],
                [16, 2],
                target_memory=ir.MemorySpace.Mat,
                mx_layout="mx_a_nd",
                span=span,
            )

    def test_rejects_vec_target_with_mx_layout(self):
        # load() defaults to Vec; mx_layout must not silently rewrite it to Mat.
        span = ir.Span.unknown()
        tensor = ir.Var("t", ir.TensorType([16, 2], DataType.FP8E8M0), span)
        with pytest.raises(Exception, match="Mat|Vec"):
            ir.op.tile.load(
                tensor,
                [0, 0],
                [16, 2],
                target_memory=ir.MemorySpace.Vec,
                mx_layout="mx_a_zz",
                span=span,
            )

    def test_mx_layout_without_target_memory_stamps_mat_kwarg(self):
        """IR create_op_call with mx_layout alone must stamp target_memory=Mat.

        DeduceTileLoadType already defaults the TileType to Mat+fractal=32; the
        Call kwargs must also carry target_memory so InferTileMemorySpace does
        not rebuild via GetImplicitTileView (ordinary Mat NZ drops fractal).
        """
        span = ir.Span.unknown()
        tensor = ir.Var("t", ir.TensorType([16, 2], DataType.FP8E8M0), span)
        offsets = ir.MakeTuple(
            [ir.ConstInt(0, DataType.INDEX, span), ir.ConstInt(0, DataType.INDEX, span)], span
        )
        shapes = ir.MakeTuple(
            [ir.ConstInt(16, DataType.INDEX, span), ir.ConstInt(2, DataType.INDEX, span)], span
        )
        call = ir.create_op_call(
            "tile.load",
            [tensor, offsets, shapes, shapes],
            {"mx_layout": "mx_a_zz"},
            span,
        )
        assert any(k == "target_memory" and v == ir.MemorySpace.Mat for k, v in call.kwargs.items())
        assert isinstance(call.type, ir.TileType)
        assert call.type.memory_space == ir.MemorySpace.Mat
        assert call.type.tile_view is not None
        assert call.type.tile_view.fractal == 32


class TestDtypeAndMemorySpace:
    def test_fp8e8m0_exists(self):
        assert DataType.FP8E8M0.get_bit() == 8
        assert DataType.FP8E8M0.to_string() == "fp8e8m0"
        assert pl.FP8E8M0 == DataType.FP8E8M0

    def test_left_right_scale_spaces(self):
        assert ir.MemorySpace.LeftScale == pl.Mem.LeftScale
        assert ir.MemorySpace.RightScale == pl.Mem.RightScale
