# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Type utilities and wrappers for PyPTO IR."""

from typing import Sequence, Union

from pypto.pypto_core import DataType
from pypto.pypto_core.ir import Expr, TensorType, TileType

from .utils import _normalize_shape

# Store the original native __init__
_native_tensor_type_init = TensorType.__init__
_native_tile_type_init = TileType.__init__


def _tensor_type_init_wrapper(
    self,
    shape: Sequence[Union[int, Expr]],
    dtype: DataType,
):
    """Wrapped __init__ for TensorType that supports integer shapes.

    Args:
        shape: Shape dimensions as a sequence of integers or Expr nodes.
               Integers are automatically converted to ConstInt(dim, DataType.INT64, Span.unknown()).
        dtype: Element data type
    """
    shape_exprs = _normalize_shape(shape)
    _native_tensor_type_init(self, shape_exprs, dtype)


def _tile_type_init_wrapper(
    self,
    shape: Sequence[Union[int, Expr]],
    dtype: DataType,
):
    """Wrapped __init__ for TileType that supports integer shapes.

    Args:
        shape: Shape dimensions as a sequence of integers or Expr nodes.
               Integers are automatically converted to ConstInt(dim, DataType.INT64, Span.unknown()).
        dtype: Element data type
    """
    shape_exprs = _normalize_shape(shape)
    _native_tile_type_init(self, shape_exprs, dtype)


# Monkey-patch the native TensorType.__init__ to support integer shapes
TensorType.__init__ = _tensor_type_init_wrapper

# Monkey-patch the native TileType.__init__ to support integer shapes
TileType.__init__ = _tile_type_init_wrapper


__all__ = ["TensorType", "TileType"]
