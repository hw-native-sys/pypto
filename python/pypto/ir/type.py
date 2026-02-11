# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Type utilities and wrappers for PyPTO IR."""

from typing import Optional, Sequence, Union

from pypto.pypto_core import DataType
from pypto.pypto_core.ir import Expr, MemRef, TensorType, TensorView, TileType, TileView

from .utils import _normalize_shape

# Store the original native __init__
_native_tensor_type_init = TensorType.__init__
_native_tile_type_init = TileType.__init__


def _tensor_type_init_wrapper(
    self,
    shape: Sequence[Union[int, Expr]],
    dtype: DataType,
    memref: Optional[MemRef] = None,
    tensor_view: Optional[TensorView] = None,
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
    shape: Sequence[Union[int, Expr]],
    dtype: DataType,
    memref: Optional[MemRef] = None,
    tile_view: Optional[TileView] = None,
):
    """Wrapped __init__ for TileType that supports integer shapes, optional MemRef and TileView.

    Args:
        shape: Shape dimensions as a sequence of integers or Expr nodes.
               Integers are automatically converted to ConstInt(dim, DataType.INT64, Span.unknown()).
        dtype: Element data type
        memref: Optional memory reference
        tile_view: Optional tile view information
    """
    shape_exprs = _normalize_shape(shape)
    if tile_view is not None and memref is None:
        raise ValueError("tile_view requires memref to be specified")
    # Always pass all 4 arguments to native constructor (memref and tile_view can be None)
    _native_tile_type_init(self, shape_exprs, dtype, memref, tile_view)


# Monkey-patch the native TensorType.__init__ to support integer shapes
TensorType.__init__ = _tensor_type_init_wrapper

# Monkey-patch the native TileType.__init__ to support integer shapes
TileType.__init__ = _tile_type_init_wrapper


__all__ = ["TensorType", "TileType"]
