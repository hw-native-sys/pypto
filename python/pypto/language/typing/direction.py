# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Parameter direction wrapper types for PyPTO Language DSL.

These types are used as AST-level markers for parameter direction annotations:
- InOut[T]: Read-write parameter
- Out[T]: Write-only output parameter

The default direction (In) requires no wrapper.

At runtime, Out[T] and InOut[T] return T unchanged via __class_getitem__.
For type checkers, they are Annotated[T, ...] aliases so Out[Tensor] resolves to Tensor.
"""

from typing import TYPE_CHECKING, TypeVar

T = TypeVar("T")

if TYPE_CHECKING:
    from typing import Annotated

    InOut = Annotated[T, "InOut"]
    Out = Annotated[T, "Out"]
else:

    class _DirectionWrapper:
        def __class_getitem__(cls, item: T) -> T:
            """Enable Wrapper[T] subscript syntax. Returns the item unchanged at runtime."""
            return item

    class InOut(_DirectionWrapper):
        """Wrapper for InOut parameter direction in type annotations.

        Usage::

            def kernel(output: pl.InOut[pl.Tensor[[16, 128], pl.FP32]]) -> ...:
        """

    class Out(_DirectionWrapper):
        """Wrapper for Out parameter direction in type annotations.

        Usage::

            def kernel(result: pl.Out[pl.Tensor[[16, 128], pl.FP32]]) -> ...:
        """


__all__ = ["InOut", "Out"]
