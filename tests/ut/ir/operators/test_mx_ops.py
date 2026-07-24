# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for MX DSL ops: matmul_mx, tquant/mx_quant, tdequant, tget_scale_addr, mx load."""

import pypto.language as pl
import pytest
from pypto import ir
from pypto.pypto_core import DataType


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

    def test_matmul_mx_acc_spec(self):
        spec = ir.get_op_memory_spec("tile.matmul_mx_acc")
        assert spec is not None
        assert spec["output_memory"] == ir.MemorySpace.Acc
        constraints = spec["input_constraints"]
        assert constraints[0] == [ir.MemorySpace.Acc]
        assert constraints[1] == [ir.MemorySpace.Left]
        assert constraints[2] == [ir.MemorySpace.LeftScale]
        assert constraints[3] == [ir.MemorySpace.Right]
        assert constraints[4] == [ir.MemorySpace.RightScale]

    def test_matmul_mx_bias_spec(self):
        spec = ir.get_op_memory_spec("tile.matmul_mx_bias")
        assert spec is not None
        assert spec["output_memory"] == ir.MemorySpace.Acc
        constraints = spec["input_constraints"]
        assert constraints[4] == [ir.MemorySpace.Bias]


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

    def test_matmul_mx_rejects_bad_scale_dtype(self):
        span = ir.Span.unknown()
        lhs = ir.Var("lhs", ir.TileType([16, 64], DataType.FP8E4M3FN), span)
        lhs_scale = ir.Var("lhs_scale", ir.TileType([16, 2], DataType.FP16), span)
        rhs = ir.Var("rhs", ir.TileType([64, 32], DataType.FP8E4M3FN), span)
        rhs_scale = ir.Var("rhs_scale", ir.TileType([2, 32], DataType.FP8E8M0), span)
        with pytest.raises(Exception, match="FP8E8M0"):
            ir.op.tile.matmul_mx(lhs, lhs_scale, rhs, rhs_scale, span)

    def test_matmul_mx_rejects_k_not_divisible_by_32(self):
        span = ir.Span.unknown()
        lhs = ir.Var("lhs", ir.TileType([16, 48], DataType.FP8E4M3FN), span)
        lhs_scale = ir.Var("lhs_scale", ir.TileType([16, 1], DataType.FP8E8M0), span)
        rhs = ir.Var("rhs", ir.TileType([48, 32], DataType.FP8E4M3FN), span)
        rhs_scale = ir.Var("rhs_scale", ir.TileType([1, 32], DataType.FP8E8M0), span)
        with pytest.raises(Exception, match="divisible by 32"):
            ir.op.tile.matmul_mx(lhs, lhs_scale, rhs, rhs_scale, span)


class TestTQuantTypes:
    def test_tquant_returns_tuple(self):
        span = ir.Span.unknown()
        src = ir.Var("src", ir.TileType([16, 64], DataType.FP32), span)
        call = ir.op.tile.tquant(src, mode="mxfp8_e4m3", span=span)
        assert isinstance(call.type, ir.TupleType)
        assert len(call.type.types) == 2
        dst, scale = call.type.types
        assert isinstance(dst, ir.TileType) and isinstance(scale, ir.TileType)
        assert dst.dtype == DataType.FP8E4M3FN
        assert scale.dtype == DataType.FP8E8M0
        assert isinstance(scale.shape[0], ir.ConstInt) and scale.shape[0].value == 16
        assert isinstance(scale.shape[1], ir.ConstInt) and scale.shape[1].value == 2

    def test_mx_quant_alias_in_dsl(self):
        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                src: pl.Tensor[[16, 64], pl.FP32],
                out_q: pl.Tensor[[16, 64], pl.FP8E4M3FN],
                out_s: pl.Tensor[[16, 2], pl.FP8E8M0],
            ):
                t: pl.Tile[[16, 64], pl.FP32] = pl.load(src, [0, 0], [16, 64])
                q, s = pl.mx_quant(t)
                pl.store(q, [0, 0], out_q)
                pl.store(s, [0, 0], out_s)

        assert "tile.tquant" in str(Program)


class TestTDequantTypes:
    def test_tdequant_type(self):
        span = ir.Span.unknown()
        src = ir.Var("src", ir.TileType([16, 64], DataType.INT8), span)
        scale = ir.Var("scale", ir.TileType([16, 1], DataType.FP32), span)
        offset = ir.Var("offset", ir.TileType([16, 1], DataType.FP32), span)
        call = ir.op.tile.tdequant(src, scale, offset, span)
        assert isinstance(call.type, ir.TileType)
        assert call.type.dtype == DataType.FP32


class TestTGetScaleAddr:
    def test_registered(self):
        spec = ir.get_op_memory_spec("tile.tget_scale_addr")
        assert spec is not None


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

    def test_mx_layout_rejects_non_e8m0(self):
        span = ir.Span.unknown()
        tensor = ir.Var("t", ir.TensorType([16, 2], DataType.FP16), span)
        with pytest.raises(Exception, match="FP8E8M0"):
            ir.op.tile.load(
                tensor,
                [0, 0],
                [16, 2],
                target_memory=ir.MemorySpace.Mat,
                mx_layout="mx_a_zz",
                span=span,
            )


class TestDtypeAndMemorySpace:
    def test_fp8e8m0_exists(self):
        assert DataType.FP8E8M0.get_bit() == 8
        assert DataType.FP8E8M0.to_string() == "fp8e8m0"
        assert pl.FP8E8M0 == DataType.FP8E8M0

    def test_left_right_scale_spaces(self):
        assert ir.MemorySpace.LeftScale == pl.Mem.LeftScale
        assert ir.MemorySpace.RightScale == pl.Mem.RightScale
