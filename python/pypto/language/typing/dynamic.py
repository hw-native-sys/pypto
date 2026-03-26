# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Dynamic shape variables for use in type annotations."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass


class DynVar:
    """Dynamic shape variable for use in type annotations.

    Creates a symbolic dimension that becomes an ir.Var node in the IR shape.

    Example:
        M = pl.dynamic("M")
        N = pl.dynamic("N")

        @pl.function
        def func(a: pl.Tensor[[M, N], pl.FP32]) -> ...:
            ...
    """

    def __init__(self, name: str) -> None:
        if not name.isidentifier():
            raise ValueError(f"DynVar name must be a valid identifier, got {name!r}")
        self.name = name
        # Lazily populated on first use by TypeResolver.  Once set, all parsers
        # that encounter this DynVar object share the same ir.Var instance,
        # ensuring structural equality across @pl.function boundaries.
        self._ir_var: Any = None

    def __repr__(self) -> str:
        return f"DynVar({self.name!r})"


def dynamic(name: str) -> DynVar:
    """Create a dynamic shape variable for type annotations.

    Args:
        name: Variable name for the dynamic dimension

    Returns:
        DynVar that can be used in shape annotations
    """
    return DynVar(name)


__all__ = ["DynVar", "dynamic"]
