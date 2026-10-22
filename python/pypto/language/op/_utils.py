# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Helpers shared by the DSL op wrappers.

Kept private: these adapt DSL-level values (``Scalar``, ``IntLike``) to the raw
``ir.Expr`` the C++ bindings accept. The tensor, tile, and distributed wrappers
all need the same coercion, so it lives here once rather than as a copy in each
namespace.
"""

from collections.abc import Sequence

from pypto.pypto_core.ir import Expr

from ..typing import IntLike, Scalar


def _normalize_intlike(seq: Sequence[IntLike]) -> list[int | Expr]:
    """Unwrap Scalar elements to Expr so the sequence matches C++ binding types."""
    return [elem.unwrap() if isinstance(elem, Scalar) else elem for elem in seq]


__all__ = ["_normalize_intlike"]
