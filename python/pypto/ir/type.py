# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Type utilities and wrappers for PyPTO IR."""

from collections.abc import Sequence

from pypto.pypto_core import DataType
from pypto.pypto_core.ir import (
    Expr,
    MemorySpace,
    MemRef,
    PadValue,
    TensorLayout,
    TensorType,
    TensorView,
    TileLayout,
    TileType,
    TileView,
)

from .utils import _normalize_shape

# Store the original native __init__
_native_tensor_type_init = TensorType.__init__
_native_tile_type_init = TileType.__init__
_native_tensor_view_init = TensorView.__init__
_native_tile_view_init = TileView.__init__

_MEMREF_NAME_PREFIX_TO_SPACE = {
    "mem_ddr_": MemorySpace.DDR,
    "mem_vec_": MemorySpace.Vec,
    "mem_mat_": MemorySpace.Mat,
    "mem_left_": MemorySpace.Left,
    "mem_right_": MemorySpace.Right,
    "mem_acc_": MemorySpace.Acc,
    "mem_bias_": MemorySpace.Bias,
}


def _infer_tile_memory_space_from_memref(memref: MemRef | None) -> MemorySpace | None:
    if memref is None:
        return None
    for prefix, memory_space in _MEMREF_NAME_PREFIX_TO_SPACE.items():
        if memref.name_hint.startswith(prefix):
            return memory_space
    return None


def _tensor_type_init_wrapper(
    self,
    shape: Sequence[int | Expr],
    dtype: DataType,
    memref: MemRef | None = None,
    tensor_view: TensorView | None = None,
):
    """Wrapped __init__ for TensorType that supports integer shapes, optional MemRef and TensorView.

    Args:
        shape: Shape dimensions as a sequence of integers or Expr nodes.
               Integers are automatically converted to ConstInt(dim, DataType.INT64, Span.unknown()).
        dtype: Element data type
        memref: Optional memory reference
        tensor_view: Optional tensor view information
    """
    shape_exprs = _normalize_shape(shape)
    # Always pass all 4 arguments to native constructor (memref and tensor_view can be None)
    _native_tensor_type_init(self, shape_exprs, dtype, memref, tensor_view)


def _tile_type_init_wrapper(
    self,
    shape: Sequence[int | Expr],
    dtype: DataType,
    memref: MemRef | None = None,
    tile_view: TileView | None = None,
    memory_space: MemorySpace | None = None,
):
    """Wrapped __init__ for TileType that supports integer shapes, optional MemRef and TileView.

    Args:
        shape: Shape dimensions as a sequence of integers or Expr nodes.
               Integers are automatically converted to ConstInt(dim, DataType.INT64, Span.unknown()).
        dtype: Element data type
        memref: Optional memory reference
        tile_view: Optional tile view information
        memory_space: Optional memory space
    """
    shape_exprs = _normalize_shape(shape)
    if memref is not None and memory_space is None:
        memory_space = _infer_tile_memory_space_from_memref(memref)
    _native_tile_type_init(self, shape_exprs, dtype, memref, tile_view, memory_space)


def _normalize_optional_shape(shape: Sequence[int | Expr] | None) -> list[Expr]:
    return _normalize_shape(shape) if shape is not None else []


def _tensor_view_init_wrapper(
    self,
    stride: Sequence[int | Expr] | None = None,
    layout: TensorLayout | None = None,
    valid_shape: Sequence[int | Expr] | None = None,
):
    """Wrapped __init__ for TensorView that supports integer stride and valid_shape.

    Args:
        stride: Stride for each dimension (int or Expr)
        layout: Tensor layout type
        valid_shape: Valid shape for each dimension (int or Expr, defaults to empty)
    """
    if stride is None and layout is None:
        _native_tensor_view_init(self)
        return
    if layout is None:
        raise ValueError("layout is required when stride is provided")
    _native_tensor_view_init(
        self,
        _normalize_optional_shape(stride),
        layout,
        _normalize_optional_shape(valid_shape),
    )


def _tile_view_init_wrapper(
    self,
    valid_shape: Sequence[int | Expr] | None = None,
    stride: Sequence[int | Expr] | None = None,
    start_offset: Expr | None = None,
    blayout: TileLayout = TileLayout.row_major,
    slayout: TileLayout = TileLayout.none_box,
    fractal: int = 512,
    pad: PadValue = PadValue.null,
):
    """Wrapped __init__ for TileView that supports integer valid_shape and stride.

    Args:
        valid_shape: Valid shape dimensions (int or Expr)
        stride: Stride for each dimension (int or Expr)
        start_offset: Starting offset
        blayout: Block layout
        slayout: Scatter layout
        fractal: Fractal size
        pad: Pad mode
    """
    if valid_shape is None and stride is None and start_offset is None:
        _native_tile_view_init(self)
        return
    if start_offset is None:
        raise ValueError("start_offset is required when valid_shape or stride is provided")
    _native_tile_view_init(
        self,
        _normalize_optional_shape(valid_shape),
        _normalize_optional_shape(stride),
        start_offset,
        blayout=blayout,
        slayout=slayout,
        fractal=fractal,
        pad=pad,
    )


# Monkey-patch the native TensorType.__init__ to support integer shapes
TensorType.__init__ = _tensor_type_init_wrapper

# Monkey-patch the native TileType.__init__ to support integer shapes
TileType.__init__ = _tile_type_init_wrapper

# Monkey-patch TensorView and TileView to support integer stride/valid_shape
TensorView.__init__ = _tensor_view_init_wrapper
TileView.__init__ = _tile_view_init_wrapper


__all__ = ["TensorType", "TileType", "TensorView", "TileView"]
