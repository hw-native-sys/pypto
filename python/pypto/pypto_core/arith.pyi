# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Type stubs for the arith submodule (arithmetic simplification utilities)."""

from pypto.pypto_core.ir import Expr

def fold_const(expr: Expr) -> Expr | None:
    """Try to constant-fold an expression.

    Accepts any BinaryExpr (Add, Sub, ...) or UnaryExpr (Neg, Abs, ...).
    Returns the folded constant result, or None if folding is not possible.

    Args:
        expr: A binary or unary expression node to fold.

    Returns:
        Folded constant expression, or None if not foldable.
    """
    ...
