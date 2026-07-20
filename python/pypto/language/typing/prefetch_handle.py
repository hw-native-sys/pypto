# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""DSL wrappers for the opaque async-prefetch handles.

Three handle classes back the ``pl.prefetch.*`` surface, each serving the two
roles :class:`pld.CommCtx` does:

* **Type annotation** — ``ctx: pl.PrefetchAsyncContext = pl.prefetch.make_context(ws)``
  declares an :class:`ir.PrefetchAsyncContextType`-valued ``Var`` in printed IR.
* **Value wrapper** — the ``pl.prefetch.*`` wrappers return these classes so
  handles flow through the DSL as typed values instead of raw ``ir.Call``
  objects. The parser's ``invoke_dsl`` unwraps via :meth:`unwrap` at the
  parser boundary.
"""

from pypto.pypto_core.ir import Expr


class _OpaqueHandle:
    """Shared behaviour for the opaque async-prefetch handle wrappers.

    Construct without arguments for an annotation-only placeholder; construct
    with ``expr=`` to wrap the IR ``Call`` returned by a ``pl.prefetch.*``
    wrapper.
    """

    def __init__(self, *, expr: Expr | None = None) -> None:
        self._expr: Expr | None = expr

    def unwrap(self) -> Expr:
        """Return the wrapped :class:`ir.Expr`.

        Raises:
            RuntimeError: If the instance was constructed without an ``expr``
                (annotation-only — never returned by a wrapper).
        """
        if self._expr is None:
            raise RuntimeError(
                f"{type(self).__name__} was constructed as an annotation placeholder, not a value wrapper"
            )
        return self._expr

    def __repr__(self) -> str:
        name = type(self).__name__
        return f"{name}(expr={self._expr})" if self._expr is not None else f"{name}()"


class PrefetchAsyncContext(_OpaqueHandle):
    """Handle to an asynchronous GM->L2 prefetch context.

    Produced by :func:`pl.prefetch.make_context`; consumed by
    :func:`pl.prefetch.async_prefetch` and :func:`pl.prefetch.session`.
    """


class AsyncEvent(_OpaqueHandle):
    """Handle to an in-flight asynchronous prefetch completion event.

    Produced by :func:`pl.prefetch.async_prefetch`; consumed by
    :func:`pl.prefetch.wait` together with the matching :class:`AsyncSession`.
    """


class AsyncSession(_OpaqueHandle):
    """Handle to the asynchronous session an :class:`AsyncEvent` belongs to.

    Produced by :func:`pl.prefetch.session`; consumed by
    :func:`pl.prefetch.wait`.
    """


__all__ = ["AsyncEvent", "AsyncSession", "PrefetchAsyncContext"]
