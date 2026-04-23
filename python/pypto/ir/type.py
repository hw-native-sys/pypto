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
    TileLayout,
    TileType,
)
from pypto.pypto_core.ir import (
    TensorView as _TensorViewBase,
)
from pypto.pypto_core.ir import (
    TileView as _TileViewBase,
)

from .utils import _normalize_expr, _normalize_shape

# Store the original native __init__
_native_tensor_type_init = TensorType.__init__
_native_tile_type_init = TileType.__init__

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
    tensor_view: _TensorViewBase | None = None,
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
    tile_view: _TileViewBase | None = None,
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


# Monkey-patch the native TensorType.__init__ to support integer shapes
TensorType.__init__ = _tensor_type_init_wrapper

# Monkey-patch the native TileType.__init__ to support integer shapes
TileType.__init__ = _tile_type_init_wrapper


def _normalize_seq(seq: Sequence[Expr | int] | None) -> list[Expr]:
    """Normalize a sequence of Expr, int, or Scalar/DynVar to a list of Expr."""
    from pypto.language.typing.scalar import Scalar  # noqa: PLC0415  # lazy: circular import

    if not seq:
        return []
    result: list[Expr] = []
    for v in seq:
        if isinstance(v, int):
            result.append(_normalize_expr(v))
        elif isinstance(v, Scalar):
            result.append(v.unwrap())
        else:
            result.append(v)
    return result


class _TensorViewMeta(type):
    """Metaclass for TensorView.

    __instancecheck__ makes isinstance(c++_instance, TensorView) return True
    for instances created by C++ code (which return _TensorViewBase, not our subclass).
    __call__ normalizes mixed int/Expr arguments before construction.
    """

    def __instancecheck__(cls, instance: object) -> bool:
        return isinstance(instance, _TensorViewBase)

    def __call__(
        cls,
        stride: Sequence[Expr | int] | None = None,
        layout: TensorLayout | None = None,
        valid_shape: Sequence[Expr | int] | None = None,
        pad: PadValue = PadValue.null,
    ) -> "_TensorViewBase":
        if stride is None and layout is None and valid_shape is None and pad == PadValue.null:
            return _TensorViewBase()
        if layout is None:
            raise ValueError("layout is required when stride, valid_shape, or pad is provided")
        return _TensorViewBase(_normalize_seq(stride), layout, _normalize_seq(valid_shape), pad)


class TensorView(metaclass=_TensorViewMeta):
    """TensorView factory: accepts Expr or int in stride/valid_shape."""


class _TileViewMeta(type):
    """Metaclass for TileView.

    __instancecheck__ makes isinstance(c++_instance, TileView) return True
    for instances created by C++ code (which return _TileViewBase, not our subclass).
    __call__ normalizes mixed int/Expr arguments before construction.
    """

    def __instancecheck__(cls, instance: object) -> bool:
        return isinstance(instance, _TileViewBase)

    def __call__(
        cls,
        valid_shape: Sequence[Expr | int] | None = None,
        stride: Sequence[Expr | int] | None = None,
        start_offset: Expr | int | None = None,
        blayout: TileLayout = TileLayout.row_major,
        slayout: TileLayout = TileLayout.none_box,
        fractal: int = 512,
        pad: PadValue = PadValue.null,
    ) -> "_TileViewBase":
        has_args = valid_shape is not None or stride is not None or start_offset is not None
        has_kwargs = (
            blayout != TileLayout.row_major
            or slayout != TileLayout.none_box
            or fractal != 512
            or pad != PadValue.null
        )
        if not has_args and not has_kwargs:
            return _TileViewBase()
        if start_offset is None:
            raise ValueError("start_offset is required")
        if isinstance(start_offset, int):
            start_offset = _normalize_expr(start_offset)
        return _TileViewBase(
            _normalize_seq(valid_shape),
            _normalize_seq(stride),
            start_offset,
            blayout,
            slayout,
            fractal,
            pad,
        )


class TileView(metaclass=_TileViewMeta):
    """TileView factory: accepts Expr or int in valid_shape/stride."""


__all__ = ["TensorType", "TileType", "TensorView", "TileView"]
