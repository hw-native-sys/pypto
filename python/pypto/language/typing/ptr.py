# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""``pl.Ptr`` — DSL wrapper for an opaque pointer handle.

Single class serving two roles, mirroring :class:`pl.Tensor` /
:class:`pl.Tile`:

* **Type annotation** — printed-IR round-trip uses ``buf: pl.Ptr =
  pl.tile.alloc(...)`` / ``buf: pl.Ptr = pld.alloc_window_buffer(...)``
  to signal that the LHS is an :class:`ir.PtrType`-valued ``Var``.
* **Value wrapper** — DSL ops that return a Ptr-typed result wrap the
  underlying :class:`ir.Expr` in this class so the DSL surface stays
  language-level (no raw ``ir.Call`` leaking out of wrappers). The
  parser's ``invoke_dsl`` unwraps via :meth:`unwrap` at the parser
  boundary, the same way it handles ``Tensor`` / ``Tile`` / ``Scalar`` /
  ``Array``.
"""

from pypto.pypto_core.ir import Expr


class Ptr:
    """DSL wrapper for an ``ir.PtrType``-valued expression.

    Construct without arguments to obtain an annotation-only placeholder
    (``buf: pl.Ptr``). Construct with ``expr=`` to wrap an IR ``Call``
    returned by an allocation op.
    """

    def __init__(self, *, expr: Expr | None = None) -> None:
        self._expr: Expr | None = expr

    def unwrap(self) -> Expr:
        """Return the wrapped :class:`ir.Expr`.

        Raises:
            RuntimeError: If the instance was constructed without an
                ``expr`` (annotation-only — never returned by a wrapper).
        """
        if self._expr is None:
            raise RuntimeError("Ptr was constructed as an annotation placeholder, not a value wrapper")
        return self._expr

    def __repr__(self) -> str:
        return f"Ptr(expr={self._expr})" if self._expr is not None else "Ptr()"


__all__ = ["Ptr"]
