# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Compile-time parameter marker for PyPTO Language DSL."""

from typing import TYPE_CHECKING, Any, TypeAlias


class ConstexprMarker:
    """Annotation marking a parameter the compiler resolves at specialization time.

    A ``pl.Scalar[dtype]`` parameter is a **runtime value**: it survives into the
    generated program, its value arrives at dispatch, and one artifact serves
    every value. ``pl.constexpr`` is the opposite end of that axis — the call
    site's value is folded into the body at every use, joins the compilation
    key, and the parameter is **absent from the generated program and from the
    dispatch ABI**. Nothing is passed for it at run time.

    Use it for a value the body needs the compiler to see: a tile shape, an
    unroll extent, or a branch the compiler should resolve away.

    A constexpr parameter is exactly a module-level constant supplied per call
    site, so it folds by the same rule and admits the same value types: ``int``,
    ``float``, ``bool``, ``str``, a ``DataType``, an enum member such as
    ``pl.Mem.Vec``, and lists of those. A value the specializer cannot render
    into source is rejected with a diagnostic naming the parameter.

    Carries no dtype: unlike ``pl.Scalar[dtype]`` it never becomes an IR
    parameter, so there is no type for the ABI to agree on.

    Examples:
        >>> import pypto.language as pl
        >>>
        >>> @pl.jit  # doctest: +SKIP
        ... def kernel(
        ...     x: pl.Tensor[[64], pl.FP32],
        ...     out: pl.Out[pl.Tensor[[64], pl.FP32]],
        ...     scale: pl.Scalar[pl.FP32],   # runtime: one artifact, any value
        ...     BLOCK: pl.constexpr,         # folded: one artifact per value
        ... ): ...
        >>>
        >>> kernel(x, out, scale=1.0, BLOCK=128)  # doctest: +SKIP
        >>> kernel(x, out, scale=2.0, BLOCK=128)  # reuses that artifact
        >>> kernel(x, out, scale=2.0, BLOCK=256)  # a second artifact
    """

    def __repr__(self) -> str:
        """Return the marker's canonical spelling."""
        return "pl.constexpr"

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Reject a call, which almost always means a missing annotation colon.

        Raises:
            TypeError: Always — the marker is an annotation, not a function.
        """
        raise TypeError(
            "pl.constexpr is an annotation, not a function. Annotate the parameter "
            "with it -- 'BLOCK: pl.constexpr' -- and pass the value at the call site."
        )


if TYPE_CHECKING:
    # The marker is a value, so a type checker would reject ``BLOCK: pl.constexpr``
    # as "variable not allowed in type expression". Present it as the permissive
    # alias instead, which is also what the annotation means: the parameter
    # accepts any compile-time constant. Same device ``pl.Out`` / ``pl.InOut``
    # use for their runtime-passthrough wrappers.
    constexpr: TypeAlias = Any
else:
    constexpr = ConstexprMarker()
    """Singleton ``ConstexprMarker`` — see the class docstring."""


__all__ = ["ConstexprMarker", "constexpr"]
