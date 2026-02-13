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
from typing import Optional

from pypto.pypto_core import DataType
from pypto.pypto_core.ir import Expr, MemRef

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
        cls, shape=None, dtype=None, expr: Optional[Expr] = None,
        _annotation_only: bool = False, memref: Optional[MemRef] = None
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
            return type.__call__(cls, real_shape, real_dtype, None, _annotation_only, memref)
        return type.__call__(cls, shape, dtype, expr, _annotation_only, memref)


class Tile(metaclass=TileMeta):
    """Tile type for PyPTO Language DSL.

    Tile represents a block in unified buffer (UB) memory. It is used for
    block-level programming with operations like load, store, add, mul, etc.

    Annotation mode (used in type hints):
        x: pl.Tile[[64, 64], pl.FP32]

    Runtime mode (wraps IR expressions):
        tile = pl.load(tensor, [0, 0], [64, 64])
        # Returns Tile wrapping the Call expression

    Runtime mode with MemRef (manual address control):
        tile = Tile(expr=call_expr, memref=my_memref)
        # The memref is stored as metadata; use bind_memref() from
        # pypto.frontend to actually attach it to the IR type.

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
        shape: Optional[Sequence[int]] = None,
        dtype: Optional[DataType] = None,
        expr: Optional[Expr] = None,
        _annotation_only: bool = False,
        memref: Optional[MemRef] = None,
    ):
        """Initialize Tile.

        Args:
            shape: Shape (for annotation mode)
            dtype: Data type (for annotation mode)
            expr: IR expression to wrap (for runtime mode)
            _annotation_only: Whether this is annotation-only mode
            memref: Optional MemRef for manual address specification.
                    When set in runtime mode, this is advisory metadata that
                    can be applied to the IR via bind_memref().
        """
        if _annotation_only:
            self.shape = shape
            self.dtype = dtype
            self._expr = None
            self._memref = None  # Not applicable in annotation mode
        elif expr is not None:
            self._expr = expr
            self.shape = None
            self.dtype = None
            self._memref = memref
        else:
            raise ValueError(
                "Tile must be initialized with either (shape, dtype) for "
                "annotations or expr for runtime wrapping"
            )

    @property
    def memref(self) -> Optional[MemRef]:
        """Get the associated MemRef, if any.

        Returns:
            The MemRef if set, None otherwise
        """
        return self._memref

    @memref.setter
    def memref(self, value: Optional[MemRef]) -> None:
        """Set the associated MemRef.

        Args:
            value: MemRef to associate, or None to clear
        """
        self._memref = value

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

    def with_memref(self, memref: MemRef) -> "Tile":
        """Create a new Tile with the given MemRef attached (fluent API).

        This does NOT modify the IR type — it only sets the Python-side
        metadata. Use bind_memref() from pypto.frontend to propagate
        the MemRef into the IR type system.

        Args:
            memref: MemRef to associate

        Returns:
            New Tile instance with the same expr but with memref set

        Raises:
            ValueError: If this is an annotation-only Tile
        """
        if self._expr is None:
            raise ValueError("Cannot set memref on annotation-only Tile")
        return Tile(expr=self._expr, memref=memref)

    @classmethod
    def __class_getitem__(cls, item: tuple[Sequence[int], DataType]) -> "Tile":
        """Support static type checkers for Tile[[shape], dtype] syntax."""
        return cls.__getitem__(item)

    def __repr__(self) -> str:
        """String representation."""
        if self._expr is not None:
            memref_str = f", memref={self._memref}" if self._memref else ""
            return f"Tile(expr={self._expr}{memref_str})"
        else:
            return f"Tile[[{self.shape}], {self.dtype}]"
