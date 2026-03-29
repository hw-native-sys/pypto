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
from typing import Any, cast, overload

from pypto.pypto_core import DataType
from pypto.pypto_core.ir import Expr, MemRef, TensorLayout


def _validate_tensor_meta_call(args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
    """Validate TensorMeta.__call__ argument structure."""
    allowed_kwargs = {"shape", "dtype", "expr", "layout", "memref", "_annotation_only"}
    unexpected = set(kwargs) - allowed_kwargs
    if unexpected:
        name = sorted(unexpected)[0]
        raise TypeError(f"Tensor() got an unexpected keyword argument '{name}'")

    if len(args) > 6:
        raise TypeError(f"Tensor() takes at most 6 positional arguments but {len(args)} were given")

    param_names = ("shape", "dtype", "expr", "layout", "memref", "_annotation_only")
    for index, name in enumerate(param_names[: len(args)]):
        if name in kwargs:
            raise TypeError(f"Tensor() got multiple values for argument '{name}'")


class TensorMeta(type):
    """Metaclass for Tensor to enable subscript notation."""

    def __getitem__(cls, item: tuple) -> "Tensor":
        """Enable Tensor[[shape], dtype], Tensor[[shape], dtype, layout_or_memref],
        and Tensor[[shape], dtype, layout, memref] syntax.

        Args:
            item: Tuple of 2, 3, or 4 elements

        Returns:
            Tensor instance with shape, dtype, and optional layout/memref
        """
        if not isinstance(item, tuple) or len(item) not in (2, 3, 4):
            raise TypeError(
                "Tensor requires [shape, dtype], [shape, dtype, layout_or_memref], "
                "or [shape, dtype, layout, memref] notation"
            )

        if len(item) == 4:
            shape, dtype, layout, memref = item
            return cls(shape, dtype, layout=layout, memref=memref, _annotation_only=True)
        if len(item) == 3:
            shape, dtype, third = item
            if isinstance(third, MemRef):
                return cls(shape, dtype, memref=third, _annotation_only=True)
            return cls(shape, dtype, layout=third, _annotation_only=True)
        shape, dtype = item
        return cls(shape, dtype, _annotation_only=True)

    @overload
    def __call__(
        cls,
        shape: Sequence[int] | None = None,
        dtype: DataType | None = None,
        expr: Expr | None = None,
        layout: "TensorLayout | None" = None,
        memref: "MemRef | None" = None,
        _annotation_only: bool = False,
    ) -> "Tensor": ...

    @overload
    def __call__(
        cls,
        shape: tuple[Any, Any],
        dtype: None = None,
        expr: None = None,
        layout: "TensorLayout | None" = None,
        memref: "MemRef | None" = None,
        _annotation_only: bool = False,
    ) -> "Tensor": ...

    def __call__(cls, *args: Any, **kwargs: Any) -> "Tensor":
        """Enable both Tensor((shape), dtype) syntax and runtime wrapping.

        Args:
            shape: Shape tuple or list (for annotation mode)
            dtype: Data type (for annotation mode)
            expr: IR expression to wrap (for runtime mode)
            layout: Optional tensor layout (ND, DN, NZ)
            memref: Optional memory reference
            _annotation_only: Internal flag for annotation-only mode

        Returns:
            Tensor instance
        """
        _validate_tensor_meta_call(args, kwargs)

        # Support metaclass instantiation for annotations
        shape = kwargs.get("shape", args[0] if len(args) > 0 else None)
        dtype = kwargs.get("dtype", args[1] if len(args) > 1 else None)
        expr = kwargs.get("expr", args[2] if len(args) > 2 else None)
        layout = kwargs.get("layout", args[3] if len(args) > 3 else None)
        memref = kwargs.get("memref", args[4] if len(args) > 4 else None)
        annotation_only = kwargs.get("_annotation_only", args[5] if len(args) > 5 else False)

        if (
            isinstance(shape, tuple)
            and len(shape) == 2
            and not isinstance(shape[0], int)
            and dtype is None
            and expr is None
        ):
            real_shape, real_dtype = shape
            return cast(
                "Tensor",
                type.__call__(cls, real_shape, real_dtype, None, layout, memref, annotation_only),
            )

        if dtype is not None and expr is None and not annotation_only:
            annotation_only = True

        return cast("Tensor", type.__call__(cls, shape, dtype, expr, layout, memref, annotation_only))


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
        memref: MemRef | None = None,
        _annotation_only: bool = False,
    ):
        """Initialize Tensor.

        Args:
            shape: Shape (for annotation mode)
            dtype: Data type (for annotation mode)
            expr: IR expression to wrap (for runtime mode)
            layout: Optional tensor layout (ND, DN, NZ)
            memref: Optional memory reference
            _annotation_only: Whether this is annotation-only mode
        """
        if _annotation_only:
            self.shape = shape
            self.dtype = dtype
            self.layout = layout
            self.memref = memref
            self._expr = None
        elif expr is not None:
            self._expr = expr
            self.shape = None
            self.dtype = None
            self.layout = None
            self.memref = None
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
        return type(cls).__getitem__(cls, item)

    def __getitem__(self, indices: Any) -> Any:
        """Subscript syntax for tensor slicing/reading (only valid inside @pl.function)."""
        raise NotImplementedError("Tensor subscript syntax is only available inside @pl.function")

    def __repr__(self) -> str:
        """String representation."""
        if self._expr is not None:
            return f"Tensor(expr={self._expr})"
        if self.layout is not None:
            return f"Tensor[[{self.shape}], {self.dtype}, {self.layout}]"
        return f"Tensor[[{self.shape}], {self.dtype}]"


__all__ = ["Tensor"]
