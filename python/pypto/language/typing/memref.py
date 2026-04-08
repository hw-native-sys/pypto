# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""MemRef wrapper type for PyPTO Language DSL.

Thin subclass of ``ir.MemRef`` that adds ``PtrType`` to the accepted
``base`` parameter so that pyright accepts ``pl.MemRef(ptr_var, 0, size)``
when ``ptr_var`` is annotated as ``pl.Ptr`` inside ``@pl.program`` code.
"""

from typing import Any, overload

from pypto.pypto_core.ir import (
    Expr,
    MemorySpace,
    PtrType,
    Span,
    Var,
)
from pypto.pypto_core.ir import (
    MemRef as _IrMemRef,
)


class MemRef(_IrMemRef):
    """DSL-level memory reference accepting PtrType-annotated variables as base.

    Identical to ``ir.MemRef`` at runtime. The only addition is that
    the ``base`` parameter also accepts ``PtrType`` so that pyright
    does not reject ``pl.MemRef(ptr_var, offset, size)`` when
    ``ptr_var`` has the annotation ``pl.Ptr``.
    """

    @overload
    def __init__(self, base: Var, byte_offset: int, size: int, span: Span = ...) -> None: ...
    @overload
    def __init__(self, base: Var, byte_offset: Expr, size: int, span: Span = ...) -> None: ...
    @overload
    def __init__(self, base: str, byte_offset: int, size: int, span: Span = ...) -> None: ...
    @overload
    def __init__(self, base: PtrType, byte_offset: int, size: int, span: Span = ...) -> None: ...
    @overload
    def __init__(self, base: PtrType, byte_offset: Expr, size: int, span: Span = ...) -> None: ...
    @overload
    def __init__(self, addr: int, size: int, id: int, span: Span = ...) -> None: ...
    @overload
    def __init__(
        self, memory_space: MemorySpace, addr: Expr | int, size: int, id: int, span: Span = ...
    ) -> None: ...
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)


__all__ = ["MemRef"]
