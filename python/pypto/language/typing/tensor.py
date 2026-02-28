# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tensor wrapper type for PyPTO Language DSL."""

from collections.abc import Sequence
from typing import Any

from pypto.pypto_core import DataType
from pypto.pypto_core.ir import Expr, TensorLayout


class TensorMeta(type):
    """Metaclass for Tensor to enable subscript notation."""

    def __getitem__(
        cls,
        item: tuple[Sequence[int], DataType] | tuple[Sequence[int], DataType, TensorLayout],
    ) -> "Tensor":
        """Enable Tensor[[shape], dtype] and Tensor[[shape], dtype, layout] syntax.

        Args:
            item: Tuple of (shape, dtype) or (shape, dtype, layout)

        Returns:
            Tensor instance with shape, dtype, and optional layout (annotation-only mode)
        """
        if not isinstance(item, tuple) or len(item) not in (2, 3):
            raise TypeError("Tensor requires [shape, dtype] or [shape, dtype, layout] notation")

        if len(item) == 3:
            shape, dtype, layout = item
            return cls(shape, dtype, layout=layout, _annotation_only=True)
        shape, dtype = item
        return cls(shape, dtype, _annotation_only=True)

    def __call__(
        cls,
        shape: Any = None,
        dtype: Any = None,
        expr: Expr | None = None,
        layout: "TensorLayout | None" = None,
        _annotation_only: bool = False,
    ) -> "Tensor":  # type: ignore[misc]
        """Enable both Tensor((shape), dtype) syntax and runtime wrapping.

        Args:
            shape: Shape tuple or list (for annotation mode)
            dtype: Data type (for annotation mode)
            expr: IR expression to wrap (for runtime mode)
            layout: Optional tensor layout (ND, DN, NZ)
            _annotation_only: Internal flag for annotation-only mode

        Returns:
            Tensor instance
        """
        # Support metaclass instantiation for annotations
        if (
            isinstance(shape, tuple)
            and len(shape) == 2
            and not isinstance(shape[0], int)
            and dtype is None
            and expr is None
        ):
            real_shape, real_dtype = shape
            return type.__call__(cls, real_shape, real_dtype, None, layout, _annotation_only)
        return type.__call__(cls, shape, dtype, expr, layout, _annotation_only)


class Tensor(metaclass=TensorMeta):
    """Tensor type for PyPTO Language DSL.

    This class serves dual purposes:
    1. Type annotation helper for function signatures
    2. Runtime wrapper around IR Expr/Call objects

    Annotation mode (used in type hints):
        x: pl.Tensor[[64, 128], pl.FP16]
        y: pl.Tensor[[64, 128], pl.FP16, pl.NZ]

    Runtime mode (wraps IR expressions):
        tensor = pl.create_tensor([64, 128], dtype=pl.FP32)
        # Returns Tensor wrapping the Call expression

    Examples:
        >>> import pypto.language as pl
        >>>
        >>> @pl.function
        ... def my_func(x: pl.Tensor[[64, 128], pl.FP16, pl.NZ]) -> pl.Tensor[[64, 128], pl.FP32]:
        ...     result: pl.Tensor[[64, 128], pl.FP32] = pl.create_tensor([64, 128], dtype=pl.FP32)
        ...     return result
    """

    def __init__(
        self,
        shape: Sequence[int] | None = None,
        dtype: DataType | None = None,
        expr: Expr | None = None,
        layout: TensorLayout | None = None,
        _annotation_only: bool = False,
    ):
        """Initialize Tensor.

        Args:
            shape: Shape (for annotation mode)
            dtype: Data type (for annotation mode)
            expr: IR expression to wrap (for runtime mode)
            layout: Optional tensor layout (ND, DN, NZ)
            _annotation_only: Whether this is annotation-only mode
        """
        if _annotation_only:
            self.shape = shape
            self.dtype = dtype
            self.layout = layout
            self._expr = None
        elif expr is not None:
            self._expr = expr
            self.shape = None
            self.dtype = None
            self.layout = None
        else:
            raise ValueError(
                "Tensor must be initialized with either (shape, dtype) for "
                "annotations or expr for runtime wrapping"
            )

    def unwrap(self) -> Expr:
        """Get underlying IR expression.

        Returns:
            The wrapped Expr/Call object

        Raises:
            ValueError: If called on an annotation-only Tensor
        """
        if self._expr is None:
            raise ValueError("Cannot unwrap annotation-only Tensor (used in type hints)")
        return self._expr

    @classmethod
    def __class_getitem__(cls, item: tuple[Sequence[int], DataType]) -> "Tensor":
        """Support static type checkers for Tensor[[shape], dtype] syntax."""
        return cls.__getitem__(item)

    def __repr__(self) -> str:
        """String representation."""
        if self._expr is not None:
            return f"Tensor(expr={self._expr})"
        if self.layout is not None:
            return f"Tensor[[{self.shape}], {self.dtype}, {self.layout}]"
        return f"Tensor[[{self.shape}], {self.dtype}]"


__all__ = ["Tensor"]
