# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""``pld.CommCtx`` — DSL wrapper for a communication-context handle.

Single class serving two roles, mirroring :class:`pl.Ptr`:

* **Type annotation** — ``ctx: pld.CommCtx = pld.get_comm_ctx(data)``
  declares an :class:`ir.CommCtxType`-valued ``Var`` in printed IR.
* **Value wrapper** — :func:`pld.get_comm_ctx` returns this class so
  ``ctx`` flows through the DSL surface as a typed handle (no raw
  ``ir.Call`` leaking from wrappers). The parser's :func:`invoke_dsl`
  unwraps via :meth:`unwrap` at the parser boundary.
"""

from pypto.pypto_core.ir import Expr


class CommCtx:
    """DSL wrapper for an ``ir.CommCtxType``-valued expression.

    Construct without arguments to obtain an annotation-only placeholder
    (``ctx: pld.CommCtx``). Construct with ``expr=`` to wrap an IR
    ``Call`` returned by :func:`pld.get_comm_ctx`.
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
            raise RuntimeError("CommCtx was constructed as an annotation placeholder, not a value wrapper")
        return self._expr

    def __repr__(self) -> str:
        return f"CommCtx(expr={self._expr})" if self._expr is not None else "CommCtx()"


__all__ = ["CommCtx"]
