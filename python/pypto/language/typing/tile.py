# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tile wrapper type for PyPTO Language DSL.

Tile represents a block in unified buffer memory, used for block-level programming.
"""

from collections.abc import Sequence

from pypto.pypto_core import DataType
from pypto.pypto_core.ir import Expr


class TileMeta(type):
    """Metaclass for Tile to enable subscript notation."""

    def __getitem__(cls, item: tuple[Sequence[int], DataType]) -> "Tile":
        """Enable Tile[[shape], dtype] syntax.

        Args:
            item: Tuple of (shape, dtype)

        Returns:
            Tile instance with shape and dtype (annotation-only mode)
        """
        if not isinstance(item, tuple) or len(item) != 2:
            raise TypeError("Tile requires [shape, dtype] notation")

        shape, dtype = item
        return cls(shape, dtype, _annotation_only=True)

    def __call__(
        cls, shape=None, dtype=None, expr: Expr | None = None, _annotation_only: bool = False
    ) -> "Tile":
        """Enable both Tile((shape), dtype) syntax and runtime wrapping."""
        if (
            isinstance(shape, tuple)
            and len(shape) == 2
            and not isinstance(shape[0], int)
            and dtype is None
            and expr is None
        ):
            real_shape, real_dtype = shape
            return type.__call__(cls, real_shape, real_dtype, None, _annotation_only)
        return type.__call__(cls, shape, dtype, expr, _annotation_only)


class Tile(metaclass=TileMeta):
    """Tile type for PyPTO Language DSL.

    Tile represents a block in unified buffer (UB) memory. It is used for
    block-level programming with operations like load, store, add, mul, etc.

    Annotation mode (used in type hints):
        x: pl.Tile[[64, 64], pl.FP32]

    Runtime mode (wraps IR expressions):
        tile = pl.load(tensor, [0, 0], [64, 64])
        # Returns Tile wrapping the Call expression

    Examples:
        >>> import pypto.language as pl
        >>>
        >>> @pl.function
        ... def my_func(input: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
        ...     tile: pl.Tile[[64, 64], pl.FP32] = pl.load(input, [0, 0], [64, 64])
        ...     result: pl.Tile[[64, 64], pl.FP32] = pl.add(tile, tile)
        ...     return pl.store(result, [0, 0], [64, 64], input)
    """

    def __init__(
        self,
        shape: Sequence[int] | None = None,
        dtype: DataType | None = None,
        expr: Expr | None = None,
        _annotation_only: bool = False,
    ):
        """Initialize Tile.

        Args:
            shape: Shape (for annotation mode)
            dtype: Data type (for annotation mode)
            expr: IR expression to wrap (for runtime mode)
            _annotation_only: Whether this is annotation-only mode
        """
        if _annotation_only:
            self.shape = shape
            self.dtype = dtype
            self._expr = None
        elif expr is not None:
            self._expr = expr
            self.shape = None
            self.dtype = None
        else:
            raise ValueError(
                "Tile must be initialized with either (shape, dtype) for "
                "annotations or expr for runtime wrapping"
            )

    def unwrap(self) -> Expr:
        """Get underlying IR expression.

        Returns:
            The wrapped Expr/Call object

        Raises:
            ValueError: If called on an annotation-only Tile
        """
        if self._expr is None:
            raise ValueError("Cannot unwrap annotation-only Tile (used in type hints)")
        return self._expr

    @classmethod
    def __class_getitem__(cls, item: tuple[Sequence[int], DataType]) -> "Tile":
        """Support static type checkers for Tile[[shape], dtype] syntax."""
        return cls.__getitem__(item)

    def __repr__(self) -> str:
        """String representation."""
        if self._expr is not None:
            return f"Tile(expr={self._expr})"
        else:
            return f"Tile[[{self.shape}], {self.dtype}]"


__all__ = ["Tile"]
