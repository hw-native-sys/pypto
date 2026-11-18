# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Annotation and value wrapper for reusable SSA pipeline completion tokens."""

from pypto.pypto_core.ir import Expr, PipeType, SyncTokenType


class SyncToken:
    """Use ``SyncToken[pipe]`` to annotate a function-local completion token."""

    def __class_getitem__(cls, item: object) -> SyncTokenType:
        """Resolve one concrete source pipeline; the function supplies ownership."""
        if not isinstance(item, PipeType):
            raise TypeError(f"SyncToken requires one PipeType argument, got {item!r}")
        return SyncTokenType(item)

    def __init__(self, expr: Expr) -> None:
        """Wrap a completion expression without allocating or consuming resources."""
        if not isinstance(expr.type, SyncTokenType):
            raise TypeError("SyncToken requires an expression with SyncTokenType")
        self._expr = expr

    def unwrap(self) -> Expr:
        """Return the underlying SSA completion expression."""
        return self._expr
