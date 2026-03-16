# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tuple wrapper type for PyPTO Language DSL.

Enables ``pl.Tuple[T1, T2, ...]`` subscript notation in type annotations,
mirroring the syntax used by ``pl.Tensor``, ``pl.Tile``, and ``pl.Scalar``.
"""

from typing import Any


class TupleMeta(type):
    """Metaclass for Tuple to enable subscript notation.

    At runtime, ``pl.Tuple[T1, T2]`` is evaluated but the result is never
    inspected — the parser works on the AST, not the runtime value.  We
    simply return a ``Tuple`` instance so the expression doesn't crash.
    """

    def __getitem__(cls, item: Any) -> "Tuple":
        """Enable Tuple[T1, T2, ...] syntax.

        Args:
            item: A single type or a tuple of types (DSL wrapper objects)

        Returns:
            Tuple annotation-only instance
        """
        types = item if isinstance(item, tuple) else (item,)
        instance = type.__call__(cls)
        instance._types = types
        return instance

    def __call__(cls, *args: Any, **kwargs: Any) -> "Tuple":
        """Support legacy pl.Tuple([T1, T2, ...]) call syntax.

        The parser works on the AST, so the return value is never inspected.
        This just prevents the expression from crashing at runtime.
        """
        instance = type.__call__(cls)
        if args:
            instance._types = tuple(args[0]) if isinstance(args[0], list) else args
        return instance


class Tuple(metaclass=TupleMeta):
    """Tuple type for PyPTO Language DSL.

    Used exclusively as a type annotation helper in function signatures.
    The parser reads the AST (not the runtime value), so this class only
    needs to make ``pl.Tuple[T1, T2, ...]`` evaluate without error.

    Annotation syntax:
        result: pl.Tuple[pl.Tensor[[64], pl.FP32], pl.Scalar[pl.INT32]]

    Examples:
        >>> import pypto.language as pl
        >>>
        >>> @pl.function
        ... def multi_return(
        ...     x: pl.Tensor[[64], pl.FP32]
        ... ) -> pl.Tuple[pl.Tensor[[64], pl.FP32], pl.Scalar[pl.INT32]]:
        ...     ...
    """

    _types: tuple[Any, ...] = ()

    def __repr__(self) -> str:
        """Return string representation."""
        inner = ", ".join(repr(t) for t in self._types)
        return f"Tuple[{inner}]"

    @classmethod
    def __class_getitem__(cls, item: Any) -> "Tuple":
        """Support static type checkers for Tuple[T1, T2, ...] syntax."""
        return type(cls).__getitem__(cls, item)


__all__ = ["Tuple"]
