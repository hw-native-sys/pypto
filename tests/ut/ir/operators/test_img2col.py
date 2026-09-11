# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""TIMG2COL's packed-layout contract and hardware operand limits."""

import pytest
from pypto import DataType, ir
from pypto.ir.op import tensor, tile


def _source(shape=(64, 32), dtype=DataType.FP16, memory=ir.MemorySpace.Mat, view=None):
    return ir.Var("src", ir.TileType(shape, dtype, tile_view=view, memory_space=memory), ir.Span.unknown())


@pytest.mark.parametrize("dtype", [DataType.FP16, DataType.BF16, DataType.FP32, DataType.INT8])
def test_img2col_result_contract(dtype):
    result = tile.img2col(
        _source(dtype=dtype), 16, 32, (16, 32), image_shape=(8, 8), kernel_size=(3, 3), padding=(1, 1, 1, 1)
    )
    assert result.op.name == ir.get_op("tile.img2col").name
    assert isinstance(result.type, ir.TileType)
    assert result.type.dtype == dtype
    assert result.type.memory_space == ir.MemorySpace.Left
    for dim, expected in zip(result.type.shape, (16, 32)):
        assert isinstance(dim, ir.ConstInt)
        assert dim.value == expected
    view = result.type.get_effective_tile_view()
    assert view.blayout == ir.TileLayout.row_major
    assert view.slayout == ir.TileLayout.row_major


@pytest.mark.parametrize(
    "src,kwargs,message",
    [
        (_source(memory=ir.MemorySpace.Vec), {}, "Mat source"),
        (_source(dtype=DataType.INT32), {}, "supports FP16"),
        (_source(shape=(63, 32)), {}, "divisible by 16"),
        (_source(shape=(64, 17)), {}, "channels divisible"),
        (
            _source(
                view=ir.TileView(
                    valid_shape=[32, 32], blayout=ir.TileLayout.col_major, slayout=ir.TileLayout.row_major
                )
            ),
            {},
            "fully valid",
        ),
        (_source(view=ir.TileView()), {}, "canonical NZ"),
        (
            _source(
                view=ir.TileView(
                    stride=[64, 1], blayout=ir.TileLayout.col_major, slayout=ir.TileLayout.row_major
                )
            ),
            {},
            "strided source",
        ),
        (_source(), {"image_shape": (4, 8)}, "equal H\\*W"),
        (_source(), {"kernel_size": (512, 1)}, "kernel_h"),
        (_source(), {"stride": (0, 1)}, "stride_h"),
        (_source(), {"padding": (256, 0, 0, 0)}, "pad_top"),
        (_source(), {"dilation": (1, 256)}, "dilation_w"),
        (_source(), {"kernel_size": (9, 9)}, "exceeds padded image"),
    ],
)
def test_img2col_rejects_invalid_geometry(src, kwargs, message):
    config = {"image_shape": (8, 8), "kernel_size": (1, 1), **kwargs}
    with pytest.raises(ValueError, match=message):
        tile.img2col(src, 0, 0, (16, 32), **config)


@pytest.mark.parametrize(
    "m,k,shape,message",
    [
        (-1, 0, (16, 16), "position"),
        (49, 0, (16, 16), "position"),
        (0, 17, (16, 16), "position"),
        (0, 1, (16, 16), "C0-aligned"),
        (0, 0, (15, 16), "aligned"),
        (0, 0, (16, 17), "aligned"),
        (0, 0, (80, 16), "exceeds unfolded image"),
    ],
)
def test_img2col_rejects_invalid_window(m, k, shape, message):
    with pytest.raises(ValueError, match=message):
        tile.img2col(_source(), m, k, shape, image_shape=(8, 8), kernel_size=(1, 1))


@pytest.mark.parametrize("level", [tensor, tile])
@pytest.mark.parametrize("dtype", [DataType.INDEX, DataType.INT64, DataType.UINT64])
def test_img2col_accepts_runtime_positions(dtype, level):
    index = ir.Var("position", ir.ScalarType(dtype), ir.Span.unknown())
    src = _tensor_source() if level is tensor else _source()
    call = level.img2col(src, index, index, (16, 16), image_shape=(8, 8), kernel_size=(1, 1))
    assert call.args[1] is index and call.args[2] is index


def _tensor_source(shape=(64, 32), dtype=DataType.FP16, view=None):
    return ir.Var("src", ir.TensorType(shape, dtype, tensor_view=view), ir.Span.unknown())


@pytest.mark.parametrize("dtype", [DataType.FP16, DataType.BF16, DataType.FP32, DataType.INT8])
def test_tensor_img2col_result_contract(dtype):
    result = tensor.img2col(
        _tensor_source(dtype=dtype),
        16,
        32,
        (16, 32),
        image_shape=(8, 8),
        kernel_size=(3, 3),
        padding=(1, 1, 1, 1),
    )
    assert result.op.name == ir.get_op("tensor.img2col").name
    assert isinstance(result.type, ir.TensorType)
    assert result.type.dtype == dtype
    for dim, expected in zip(result.type.shape, (16, 32)):
        assert isinstance(dim, ir.ConstInt)
        assert dim.value == expected


@pytest.mark.parametrize(
    "src,kwargs,message",
    [
        (_source(), {}, "2D source tensor"),
        (_tensor_source(shape=(64,)), {}, "2D source tensor"),
        (_tensor_source(dtype=DataType.INT32), {}, "supports FP16"),
        (_tensor_source(shape=(63, 32)), {}, "divisible by 16"),
        (_tensor_source(shape=(64, 17)), {}, "channels divisible"),
        (
            _tensor_source(view=ir.TensorView(layout=ir.TensorLayout.ND, valid_shape=[32, 32])),
            {},
            "fully valid",
        ),
        (_tensor_source(), {"image_shape": (4, 8)}, "equal H\\*W"),
        (_tensor_source(), {"stride": (0, 1)}, "stride_h"),
        (_tensor_source(), {"kernel_size": (9, 9)}, "exceeds padded image"),
    ],
)
def test_tensor_img2col_rejects_invalid_source(src, kwargs, message):
    with pytest.raises(ValueError, match=message):
        tensor.img2col(src, 0, 0, (16, 32), **{"image_shape": (8, 8), "kernel_size": (1, 1), **kwargs})


@pytest.mark.parametrize("level", [tensor, tile])
@pytest.mark.parametrize(
    "geometry", [(8,), (8, True), (8, ir.Var("w", ir.ScalarType(DataType.INDEX), ir.Span.unknown()))]
)
def test_img2col_rejects_nonstatic_geometry(level, geometry):
    src = _tensor_source() if level is tensor else _source()
    with pytest.raises(ValueError, match="compile-time integer"):
        level.img2col(src, 0, 0, (16, 32), image_shape=geometry, kernel_size=(1, 1))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
