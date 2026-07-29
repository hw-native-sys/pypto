# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for MX scale load layouts and LeftScale/RightScale spaces."""

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
