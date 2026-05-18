# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Shared helpers for ``pld.*`` DSL wrappers."""

from collections.abc import Sequence
from typing import Any

from pypto.language.typing import IntLike, Scalar
from pypto.pypto_core.ir import Expr


def _unwrap(value: Any) -> Any:
    """Unwrap a DSL wrapper (Tensor / Tile / Scalar / Ptr / CommCtx / ...) to ``ir.Expr``.

    Falls through unchanged for raw ``ir.Expr`` and primitive ``int`` /
    ``float`` values (which downstream IR builders normalise to ``ConstInt`` /
    ``ConstFloat``).
    """
    if hasattr(value, "unwrap"):
        return value.unwrap()
    return value


def _normalize_intlike(seq: Sequence[IntLike]) -> list[int | Expr]:
    """Unwrap Scalar elements to Expr so the sequence matches C++ binding types."""
    return [elem.unwrap() if isinstance(elem, Scalar) else elem for elem in seq]


__all__ = ["_normalize_intlike", "_unwrap"]
