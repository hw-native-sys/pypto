# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Static register annotations and runtime wrappers for logical Reg values."""

from typing import cast

from pypto.pypto_core import DataType, ir
from pypto.pypto_core.ir import RegKind, RegType


class Reg:
    """Use ``Reg[lanes, dtype, kind]``; the default kind is VecReg."""

    def __class_getitem__(cls, item: object) -> RegType:
        if not isinstance(item, tuple):
            raise TypeError("Reg requires [lanes, dtype, optional kind] notation")
        fields = cast(tuple[object, ...], item)
        if len(fields) not in (2, 3):
            raise TypeError("Reg requires [lanes, dtype, optional kind] notation")
        shape, dtype = fields[:2]
        kind = fields[2] if len(fields) == 3 else RegKind.VecReg
        if not isinstance(shape, int) or isinstance(shape, bool):
            raise TypeError("Reg lanes must be a static positive integer")
        if not isinstance(dtype, DataType) or not isinstance(kind, RegKind):
            raise TypeError("Reg requires a DataType and a RegKind")
        return RegType(shape, dtype, kind)

    def __init__(self, expr: object) -> None:
        if not isinstance(expr, ir.Expr) or not isinstance(expr.type, RegType):
            raise TypeError("Reg requires an expression with RegType")
        self._expr = expr

    def unwrap(self) -> ir.Expr:
        """Return the underlying register expression."""
        return self._expr
