# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""DSL API helpers for writing IR functions.

NOTE: This module re-exports from pypto.language for backwards compatibility.
New code should use pypto.language directly.
"""

from typing import Any, Sequence, Tuple

from pypto.ir import DataType

# Import and re-export from pypto.language for backwards compatibility
from pypto.language.dsl_api import RangeIterator, range, yeild


# Keep the old Tensor class for backwards compatibility
# This is now just a type annotation helper, the actual runtime wrapper is in pypto.language.Tensor
class TensorMeta(type):
    """Metaclass for Tensor to enable subscript notation."""

    def __getitem__(cls, item: Tuple[Sequence[int], DataType]) -> "Tensor":
        """Enable Tensor[[shape], dtype] syntax (recommended).

        Args:
            item: Tuple of (shape, dtype)

        Returns:
            Tensor instance with shape and dtype
        """
        if not isinstance(item, tuple) or len(item) != 2:
            raise TypeError("Tensor requires [shape, dtype] notation")

        shape, dtype = item
        return cls(shape, dtype)

    def __call__(cls, shape: Sequence[int], dtype: Any) -> "Tensor":
        """Enable Tensor((shape), dtype) syntax (legacy).

        Args:
            shape: Shape tuple or list
            dtype: Data type (e.g., pl.FP16)

        Returns:
            Tensor instance with shape and dtype
        """
        # Support both call and metaclass instantiation
        if isinstance(shape, tuple) and len(shape) == 2 and not isinstance(shape[0], int):
            # This might be __call__ from metaclass with (shape, dtype) already unpacked
            # In that case, shape is actually (real_shape, real_dtype)
            return type.__call__(cls, shape, dtype)
        return type.__call__(cls, shape, dtype)


class Tensor(metaclass=TensorMeta):
    """Type annotation helper for tensor types.

    This is used in function signatures to specify tensor parameter types.

    Supports two syntaxes:
    - Recommended: pl.Tensor[[64, 128], pl.FP16]
    - Legacy: pl.Tensor((64, 128), pl.FP16)

    Examples:
        >>> @pl.function
        ... def my_func(x: pl.Tensor[[64, 128], pl.FP16]) -> pl.Tensor[[64, 128], pl.FP32]:
        ...     pass
        >>>
        >>> # Legacy syntax also supported
        >>> @pl.function
        ... def legacy(x: pl.Tensor((64, 128), pl.FP16)) -> pl.Tensor((64, 128), pl.FP32):
        ...     pass
    """

    def __init__(self, shape: Sequence[int], dtype: DataType):
        """Initialize tensor type annotation.

        Args:
            shape: Shape (list or tuple)
            dtype: Data type (e.g., pl.FP16)
        """
        self.shape = shape
        self.dtype = dtype

    @classmethod
    def __class_getitem__(cls, item: Tuple[Sequence[int], DataType]) -> "Tensor":
        """Support static type checkers for Tensor[[shape], dtype] syntax."""
        return cls.__getitem__(item)


__all__ = ["range", "yeild", "Tensor", "RangeIterator"]
