# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Scalar wrapper type for PyPTO Language DSL."""

from typing import Any, Optional

from pypto.pypto_core import DataType
from pypto.pypto_core.ir import Expr


class ScalarMeta(type):
    """Metaclass for Scalar to enable subscript notation."""

    def __getitem__(cls, dtype: DataType) -> "Scalar":
        """Enable Scalar[dtype] syntax (recommended).

        Args:
            dtype: Data type

        Returns:
            Scalar instance with dtype (annotation-only mode)
        """
        return cls(dtype, _annotation_only=True)

    def __call__(
        cls, dtype: Any = None, expr: Optional[Expr] = None, _annotation_only: bool = False
    ) -> "Scalar":  # type: ignore[misc]
        """Enable both Scalar(dtype) syntax and runtime wrapping.

        Args:
            dtype: Data type (for annotation mode)
            expr: IR expression to wrap (for runtime mode)
            _annotation_only: Internal flag for annotation-only mode

        Returns:
            Scalar instance
        """
        # When called with just dtype (legacy notation), treat it as annotation mode
        if dtype is not None and expr is None and not _annotation_only:
            _annotation_only = True
        return type.__call__(cls, dtype, expr, _annotation_only)


class Scalar(metaclass=ScalarMeta):
    """Scalar type for PyPTO Language DSL.

    This class serves dual purposes:
    1. Type annotation helper for function signatures
    2. Runtime wrapper around IR Expr/Call objects

    Annotation mode (used in type hints):
        x: pl.Scalar[pl.FP32]
        count: pl.Scalar[pl.INT32]

    Runtime mode (wraps IR expressions):
        scalar_value = pl.op.scalar.create(3.14, dtype=pl.FP32)
        # Returns Scalar wrapping the Call expression

    Examples:
        >>> import pypto.language as pl
        >>>
        >>> @pl.function
        ... def add_scalar(
        ...     x: pl.Tensor[[64], pl.FP32],
        ...     scalar: pl.Scalar[pl.FP32]
        ... ) -> pl.Tensor[[64], pl.FP32]:
        ...     result: pl.Tensor[[64], pl.FP32] = pl.op.add(x, scalar)
        ...     return result
    """

    def __init__(
        self,
        dtype: Optional[DataType] = None,
        expr: Optional[Expr] = None,
        _annotation_only: bool = False,
    ):
        """Initialize Scalar.

        Args:
            dtype: Data type (for annotation mode)
            expr: IR expression to wrap (for runtime mode)
            _annotation_only: Internal flag for annotation-only mode

        Raises:
            ValueError: If neither dtype nor expr is provided
        """
        if _annotation_only:
            # Annotation mode: store dtype for type checking
            if dtype is None:
                raise ValueError("dtype is required for annotation mode")
            self.dtype = dtype
            self.expr = None
            self._annotation_only = True
        elif expr is not None:
            # Runtime mode: wrap IR expression
            self.expr = expr
            self.dtype = None
            self._annotation_only = False
        else:
            raise ValueError("Either dtype (for annotation) or expr (for runtime) must be provided")

    def unwrap(self) -> Expr:
        """Unwrap to get the underlying IR expression.

        Returns:
            The wrapped IR expression

        Raises:
            RuntimeError: If this is an annotation-only instance
        """
        if self._annotation_only:
            raise RuntimeError("Cannot unwrap annotation-only Scalar")
        if self.expr is None:
            raise RuntimeError("No expression to unwrap")
        return self.expr

    def __repr__(self) -> str:
        """Return string representation."""
        if self._annotation_only:
            return f"Scalar[{self.dtype}]"
        return f"Scalar(expr={self.expr})"

    @classmethod
    def __class_getitem__(cls, item: DataType) -> "Scalar":
        """Support static type checkers for Scalar[dtype] syntax."""
        return cls.__getitem__(item)


__all__ = ["Scalar"]
