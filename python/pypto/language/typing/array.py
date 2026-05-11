# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Array wrapper type for PyPTO Language DSL.

An ``Array`` is a small fixed-size homogeneous 1-D array that lives on the
on-core scalar register file / C stack — distinct from ``Tensor`` (GM/DDR)
and ``Tile`` (vector / cube hardware state).

Writes are SSA-functional: ``arr[i] = v`` desugars to
``arr = pl.array.update_element(arr, i, v)``, which returns a *new* SSA value
of ``ArrayType``. Codegen later lowers a chain of update_element Calls back
to in-place C-stack mutation.

Typical usage:

    route_dst = pl.array.create(N_ROUTES, pl.INT32)
    route_dst[r] = dst           # __setitem__ -> array.update_element (functional)
    cd = route_dst[j]            # __getitem__ -> array.get_element
"""

from typing import TYPE_CHECKING, Any, cast

from pypto.pypto_core import DataType
from pypto.pypto_core.ir import Expr

if TYPE_CHECKING:
    from pypto.language.typing.scalar import Scalar


def _validate_array_meta_call(args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
    allowed_kwargs = {"extent", "dtype", "expr", "_annotation_only"}
    unexpected = set(kwargs) - allowed_kwargs
    if unexpected:
        name = sorted(unexpected)[0]
        raise TypeError(f"Array() got an unexpected keyword argument '{name}'")

    if len(args) > 4:
        raise TypeError(f"Array() takes at most 4 positional arguments but {len(args)} were given")

    param_names = ("extent", "dtype", "expr", "_annotation_only")
    for index, name in enumerate(param_names[: len(args)]):
        if name in kwargs:
            raise TypeError(f"Array() got multiple values for argument '{name}'")


class ArrayMeta(type):
    """Metaclass enabling ``pl.Array[extent, dtype]`` subscript notation."""

    def __getitem__(cls, item: tuple) -> "Array":
        if not isinstance(item, tuple) or len(item) != 2:
            raise TypeError("Array requires [extent, dtype] notation")
        extent, dtype = item
        return cls(extent, dtype, _annotation_only=True)

    def __call__(cls, *args: Any, **kwargs: Any) -> "Array":
        _validate_array_meta_call(args, kwargs)

        extent = kwargs.get("extent", args[0] if len(args) > 0 else None)
        dtype = kwargs.get("dtype", args[1] if len(args) > 1 else None)
        expr = kwargs.get("expr", args[2] if len(args) > 2 else None)
        annotation_only = kwargs.get("_annotation_only", args[3] if len(args) > 3 else False)

        # Bare `pl.Array(N, dtype)` is treated as annotation by default to mirror
        # Tensor/Tile DSL ergonomics.
        if dtype is not None and expr is None and not annotation_only:
            annotation_only = True

        return cast(
            "Array",
            type.__call__(cls, extent, dtype, expr, annotation_only),
        )


class Array(metaclass=ArrayMeta):
    """On-core array wrapper.

    Annotation mode (used in type hints — rare; Arrays don't cross function
    boundaries in v1, but the annotation is available for clarity and
    forward-compat)::

        arr: pl.Array[16, pl.INT32]

    Runtime mode (the common path)::

        arr = pl.array.create(16, pl.INT32)
        arr[i] = value       # -> array.update_element (functional, rebinds arr)
        x = arr[i]           # -> array.get_element
    """

    def __init__(
        self,
        extent: int | None = None,
        dtype: DataType | None = None,
        expr: Expr | None = None,
        _annotation_only: bool = False,
    ) -> None:
        if _annotation_only:
            self.extent = extent
            self.dtype = dtype
            self._expr: Expr | None = None
        elif expr is not None:
            self._expr = expr
            self.extent = None
            self.dtype = None
        else:
            raise ValueError(
                "Array must be initialized with either (extent, dtype) for "
                "annotations or expr for runtime wrapping"
            )

    def unwrap(self) -> Expr:
        if self._expr is None:
            raise ValueError("Cannot unwrap annotation-only Array (used in type hints)")
        return self._expr

    # --- Indexing sugar -----------------------------------------------------

    def __getitem__(self, index: "int | Expr | Scalar") -> "Scalar":  # noqa: F821
        from pypto.language.op import array as _array_dsl  # noqa: PLC0415

        return _array_dsl.get_element(self, index)

    def __setitem__(self, index: "int | Expr | Scalar", value: "int | Expr | Scalar") -> None:  # noqa: F821
        # Note: this assigns the result back to self._expr so subsequent reads
        # see the updated SSA value. The parser's surface-level sugar handles
        # the variable-name rebinding for code inside @pl.function — this
        # path is for direct DSL usage (tests, REPL-style construction).
        from pypto.language.op import array as _array_dsl  # noqa: PLC0415

        updated = _array_dsl.update_element(self, index, value)
        self._expr = updated.unwrap()
